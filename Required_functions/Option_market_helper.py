import pandas as pd
import os
import numpy as np
from py_vollib_vectorized import implied_volatility
import py_vollib

class OptionMarket:
    def __init__(self, option_data_prefix, stock_price_data_prefix,spot_curve_df,forward_curve_df,option_right="both"):
        """
        Initializes the OptionMarket class with option data, par yield curve, and stock price data.

        Parameters:
        option_data_prefix (str): Prefix for the option data files.
        par_yield_curve_prefix (str): Prefix for the yield curve data files.
        stock_price_data_prefix (str): Prefix for the csv file containing stock price data.
        """
        self.option_data_prefix = option_data_prefix  # Store the prefix for option data files
        self.stock_price_data_prefix = stock_price_data_prefix  # Store stock price data
        
        self.option_right=option_right
        print(f'Option rights consider {option_right}')
        self.spot_curve_df=spot_curve_df
        self.forward_curve_df=forward_curve_df
        self.option_df = self.load_option_data()  # Load the option data
        if stock_price_data_prefix!="option_dx":
            self.stock_df = self.load_stock_price_data()
        
        self.original_market_df = self.merge_option_stock_yields()
        
    def load_option_data(self):
        """
        Loads option data from CSV files in the current directory.

        Returns:
        DataFrame containing combined option data.
        """
        option_data_prefix=self.option_data_prefix
        try:
            current_directory_str = os.getcwd()  # Current working directory

            # List all files starting with the option_data_prefix
            options_csv_files = [f for f in os.listdir(current_directory_str) if f.startswith(option_data_prefix)]

            # Create full paths for the files
            options_filepaths_list = [os.path.join(current_directory_str, f) for f in options_csv_files]

            # Read the files into a DataFrame
            option_df = pd.concat(map(pd.read_csv, options_filepaths_list), ignore_index=True)
            print(f"We have successfully loaded and merged all CSV files beginning with '{option_data_prefix}' in your working directory. Storing in object.")
            return option_df
        except Exception as e:
            print(f"An error occurred while loading option data: {e}. Check that files beginning with {option_data_prefix} are in {current_directory_str}")
    # def load_fed_sheet(self):
    #     fed_sheet_name=self.par_yield_curve_prefix
    #     fed_sheet=fed_yield_curve_maker(fed_sheet_name)
        
    #     yield_curve = {}
    #     group=fed_sheet.groupby(["Date"])
    #     group.apply(lambda x: yield_curve.update(compute_yields(x,plotting=False)))
    #     return yield_curve
    def load_stock_price_data(self):
        """
        Loads the stock data into a dataframe.
        Returns:
        Dataframe containing the stock price.
        """
        stock_price_data_prefix=self.stock_price_data_prefix
        current_directory_str = os.getcwd()  # Current working directory
            
            # Attempt to load stock data files
        try:
                # List all files starting with the stock_price_data_prefix
            stock_csv_files = [f for f in os.listdir(current_directory_str) if f.startswith(stock_price_data_prefix)]
                
            if not stock_csv_files:
                raise FileNotFoundError(f"No files found starting with '{stock_price_data_prefix}'.")
    
                # Create full paths for the files and read them into a DataFrame
            stock_filepaths_list = [os.path.join(current_directory_str, f) for f in stock_csv_files]
            stock_df = pd.concat(map(pd.read_csv, stock_filepaths_list), ignore_index=True)
            
            stock_header=stock_df.columns
            if any("ate" in column for column in stock_header): #finds date column name
                date_column_name = list(filter(lambda word: "ate" in word, stock_header))
            else:
                raise FileNotFoundError(f"The stock data file must contain a column labeled date. We could not find it")


                # Convert 'dates' column to datetime
            stock_df['Date_column'] = pd.to_datetime(stock_df[date_column_name[0]])
            print("--" * 20)  # Print a line of dashes for separation
            print(f"We have successfully loaded the stock data files beginning with '{stock_price_data_prefix}' in your working directory.")

                # Check for the required 'price' column
            if 'price' not in stock_df.columns:
                raise ValueError("The stock data file must contain a column labeled 'price'.")
            if 'price' in stock_df.columns:
                stock_df=stock_df[['Date_column','price']]
                
                all_dates = pd.date_range(start=stock_df['Date_column'].min(), 
                                          end=stock_df['Date_column'].max(), freq='D')
        
                # Find missing dates (i.e., days that should have data but don't)
                missing_dates = all_dates.difference(stock_df['Date_column'])
        
                # Reindex the DataFrame to include missing dates, then fill with the previous day's yield curve
                stock_df = stock_df.set_index('Date_column').reindex(all_dates).ffill().reset_index()
        
                # Rename index column back to Date
                stock_df = stock_df.rename(columns={'index': 'Date_column'})
            return stock_df
    
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return None  # Return None if there's an error
    
        except Exception as e:
            print(f"An error occurred while loading the stock data: {e}. Check that files beginning with '{stock_price_data_prefix}' are in {current_directory_str}")
            return None  # Return None if there's a different error

    def merge_option_stock_yields(self):
        """
        Merges the stock data and the option dataframe together, and appends the risk free rate, as calibrated from niegel-svenson model to another column beside the maturity.
        Additionally, only keeps OTM options by replacing ITM calls with OTM puts using the put-call parity.
        Returns:
        Dataframe containing options, stock data and risk free rate.
        
        """
        print("--"*20)
        print("We are now merging option, yield curve and stock data")
        option_df=self.load_option_data()
        #stock_df=self.load_stock_price_data()
        #self.stock_df=stock_df
        if self.stock_price_data_prefix!="option_dx":
            stock_df=self.stock_df
        spot_curve_df=self.spot_curve_df
        forward_curve_df=self.forward_curve_df
        option_right=self.option_right
        #yield_curve_df=self.load_par_yield_curve()
        #self.yield_curve=yield_curve_df
        option_df.columns = option_df.columns.str.strip()
        header_list=option_df.columns.tolist()
        
        if "[BASE_CURRENCY]" in header_list:
            source_identification="Options_dx_bitcoin_eod_quotes"
            option_df=clean_btc_optionsdx(option_df,stock_df,spot_curve_df,forward_curve_df,option_right)

            print("--" * 20)  # Print a line of dashes for separation
            print("We believe this data is end of day quotes for bitcoin options from Options_DX")
        if "cp_flag" in header_list:
            option_df=clean_optionmetrics(option_df,stock_df,spot_curve_df,forward_curve_df,option_right)
            print("Options_metrics")
        # if "[QUOTE_UNIXTIME]" in header_list:
        #     option_df,stock_df=clean_options_dx(option_df,spot_curve_df,forward_curve_df)
        #     self.stock_df=stock_df
        
        
        
        
        option_df=option_df.reset_index(drop=True)
        #self.original_market_df=option_df
        return option_df
    def set_master_dataframe(self,volume=-1,maturity=[-1,5000],moneyness=[-1000,1000],rolling=False,date="all"):
        """
        

        Parameters
        ----------
        volume : float
            Removes all options with volume less than or equal to this number.
        maturity : list
            removes all options with maturity less than maturity[0] or greater than maturity[1]. If list has only 1 entry, only options with maturity exactly equal to maturity[0] are kept. Maturity is in DAYs.
            i.e maturity=[1,2] keeps all options with maturities of 1 or 2 days.
        Rolling: float
            if True Keeps all options with the maturity closest to float.
            
        Returns
        -------
        None.

        """
        original_dataframe=self.original_market_df.copy()
        
        original_dataframe=original_dataframe[original_dataframe["volume"]>volume]
        original_dataframe=original_dataframe[original_dataframe["bid_size"]>volume]
        original_dataframe=original_dataframe[original_dataframe["offer_size"]>volume]

        ###
        original_dataframe=original_dataframe[(original_dataframe["rounded_maturity"]>=maturity[0]/365) & (original_dataframe["rounded_maturity"]<=maturity[1]/365)]
        ##
        original_dataframe=original_dataframe[(original_dataframe["moneyness"]>=moneyness[0]) & (original_dataframe["moneyness"]<=moneyness[1])]
        ##
        if rolling != False:
            date_group=original_dataframe.groupby(["date"])
            original_dataframe = date_group.apply(lambda x: rolling_window(x,rolling))
        
        original_dataframe=original_dataframe.reset_index(drop=True)
        if date != "all":
            start_date = pd.to_datetime(date[0])
            end_date = pd.to_datetime(date[1])
            original_dataframe=original_dataframe[ (original_dataframe["qoute_dt"]>=start_date) & (original_dataframe["qoute_dt"]<=end_date)]
        self.master_dataframe=original_dataframe
        return None
def clean_btc_optionsdx(option_df,stock_df,spot_curve_df,forward_curve_df,option_right):
    """
    Description of function: Merges stock data, yield data and option data into one dataframe, and does basic data cleaning.

    Parameters
    ----------
    option_df : dataframe
        dataframe of options
    stock_df : dataframe
        dataframe of stock prices.
    yield_curve_df : dataframe
        dataframe of yield curve.

    Returns
    option_df, a dataframe containing merged information and cleaning.
    None.

    """
    ###extracting header list and removing brackets
    header_list=option_df.columns.tolist()
    header_list = [entry.replace("[", "").replace("]", "") for entry in header_list]
    option_df.columns=header_list
    
    ##Renaming column headers to a standard convention, and creating date-time values        
    option_df=option_df.loc[:,["QUOTE_DATE","EXPIRY_DATE","ASK_PRICE","BID_PRICE","STRIKE",'MARK_IV',"VOLUME","BID_SIZE","ASK_SIZE","UNDERLYING_INDEX","UNDERLYING_PRICE","OPTION_RIGHT"]]
    option_df.columns=["date","exdate","best_offer","best_bid","strike","mid_iv","volume","bid_size","offer_size","underlying_future","underlying_price","option_right"]
    option_df['option_right'] = option_df['option_right'].str.strip()
    option_df['date'] = option_df['date'].str.strip()
    option_df['exdate'] = option_df['exdate'].str.strip()
    option_df['qoute_dt'] = pd.to_datetime(option_df['date'])# + pd.to_timedelta(8, unit='h') #qoute 8:00UTC
    option_df['expiry_dt'] = pd.to_datetime(option_df['exdate'])# + pd.to_timedelta(22, unit='h') #Expires 10PM central
    
   
    ###Scaling mid_iv to decimal
    option_df["mid_iv"]=option_df["mid_iv"]/100        
    ####Pasting stock data into option frame, and converts the options from BTC into US dollars.
    date_group=option_df.groupby(["date"])
    option_df = option_df.merge(stock_df, how='left', left_on='qoute_dt', right_on='Date_column')
    option_df = option_df.rename(columns={'price': 'stock_price'})
    
    #Adjusting datetimevalues #we use 0.001
    option_df=option_df[option_df["best_offer"]>0.001]
    
    option_df["best_offer"]=option_df["best_offer"]*option_df['stock_price']
    option_df["best_bid"]=option_df["best_bid"]*option_df['stock_price']
            
    ####Addressing deribit future issues See helper function
    date_group=option_df.groupby(['date','exdate'])
    print("--" * 20)  # Print a line of dashes for separation
    print("We are averaging bitcoin futures")    
    option_df=date_group.apply(lambda x: average_futures(x))
    option_df=option_df.reset_index(drop=True)

    ####Cleaning volume. Any missing data is replaced with zero. Leading space removed.
    option_df["volume"]=option_df["volume"].astype(str)
    option_df['volume'] =option_df['volume'].replace(' ', "0")
    option_df["volume"] =option_df["volume"].str.lstrip()
    option_df["volume"]=option_df["volume"].astype(float)
    option_df=option_df.reset_index(drop=True)
        
    #### Converting "calls" and "puts" to "c" and "p" respectively.
    option_df['option_right'] = option_df['option_right'].replace({'call': 'c', 'put': 'p'})
    if option_right=="calls":
        option_df=option_df[option_df["option_right"]=="c"]
    if option_right=="puts":
        option_df=option_df[option_df["option_right"]=="p"]
###############################################################General Cleaning

    ###Creating maturity in year fraction. For using black scholes formula I replace 0 with 0.001 for numerical reasons.
    option_df['maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365
    option_df['maturity']=option_df['maturity']+(20/24)/365 #expires 10pm, qouted 2am.
    #option_df['maturity'] =option_df['maturity'].replace(0, 0.001)
    ##rounded maturity for grouping purposes
    option_df['rounded_maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365
    ###Creating mid price. Removing any instance in which the mid price is negative
    option_df['mid_price']=(option_df['best_bid']+option_df['best_offer'])/2
    option_df=option_df[option_df['mid_price']>0]
    ##Creating Log-moneyness column
    option_df.loc[option_df['option_right']=='p','log moneyness'] = np.log(option_df['strike']/option_df['stock_price'])
    option_df.loc[option_df['option_right']=='c','log moneyness'] = np.log(option_df['stock_price']/option_df['strike'])
    
    ###################################################### Yield Curve Interpolation with Cubic Splines
    #######
    #option_df["risk_free_rate"]=0
    #option_df = attach_risk_free_rate(option_df, yield_curve)
    option_df=pasting_yields_option_df(option_df, spot_curve_df, "risk_free_rate")
    option_df=pasting_yields_option_df(option_df, forward_curve_df, "forward_risk_free_rate")
    ######
    # ##merging yield curve onto main option dataframe
    # option_df = option_df.merge(yield_curve_df, how='left', left_on='qoute_dt', right_on='Date')
    # option_df.drop(columns=['Date'], inplace=True)
    # date_group=option_df.groupby(['date'])
    # yield_curve_headers=yield_curve_df.columns.tolist()[1:]
    
    # print("--" * 20)  # Print a line of dashes for separation
    # print("We are interpolating the yield curve to match your options. This may take some time.")
    # ###Performing Yield Curve interpolation
    # option_df=option_df.reset_index(drop=True)
    # option_df=date_group.apply(lambda x: paste_rate(x,yield_curve_headers)) 
    # print("We are done interpolating Yield curve")
    # option_df=option_df.reset_index(drop=True)
    
    
    ################################################   Use only OTM options. Converts ITM puts to (OTM) calls via put-call parity. Remove ITM calls. 
    print("--" * 20)  # Print a line of dashes for separation
    print("Converting puts to calls. If both are present we use only OTM options.")
    date_group=option_df.groupby(['date','exdate'])
    option_df=date_group.apply(lambda x: use_only_OTM(x))
    option_df=option_df.reset_index(drop=True)

    ##############################################  Aggregates duplicate option prices if they exist
    #Aggregates duplicates mid prices *See remove_duplicates function
    
    date_group=option_df.groupby(['date','exdate'])
    print("--" * 20)  # Print a line of dashes for separation
    print("We are Aggregating duplicate option entries")    
    option_df=date_group.apply(lambda x: remove_duplicates(x))
    option_df=option_df.reset_index(drop=True)
    option_df['option_right'] = option_df['option_right'].replace({'call parity': 'c'})
    ###
    # mid_iv=py_vollib.black.implied_volatility.implied_volatility(option_df["mid_price"], 
    #                                                              option_df["stock_price"], option_df["strike"], option_df["risk_free_rate"], option_df["maturity"], 'c', return_as='numpy')
    # option_df["mid_iv"]=mid_iv
    
    
    option_df["moneyness"]=option_df["strike"]/option_df['stock_price']-1
    option_df = option_df.dropna()
    option_df['date'] = pd.to_datetime(option_df['date']).dt.strftime('%Y-%m-%d')
    option_df['exdate'] = pd.to_datetime(option_df['exdate']).dt.strftime('%Y-%m-%d')
    ####keep only neccessary columns
    option_df=option_df[["date","exdate","maturity","rounded_maturity","risk_free_rate","forward_risk_free_rate","strike","best_bid","mid_price","best_offer","mid_iv","underlying_price","stock_price",
                        "volume","offer_size","bid_size","option_right","moneyness","qoute_dt","expiry_dt"]]
    option_df=option_df.reset_index(drop=True)
    ##Computes BS IV

    
    return option_df  
def clean_optionmetrics(option_df,stock_df,spot_curve_df,forward_curve_df,option_right):
    """
    Description of function: To clean other option_metrics data. Coming soon.

    Parameters
    ----------
    option_df : TYPE
        DESCRIPTION.
    stock_df : TYPE
        DESCRIPTION.
    yield_curve_df : TYPE
        DESCRIPTION.

    Yields
    ------
    option_df : TYPE
        DESCRIPTION.

    """
    header_list=option_df.columns.tolist()
    try:
        option_df=option_df[["date","exdate","best_offer","best_bid","strike_price","cp_flag","volume"]]
    except Exception as e:
        print("Missing information. Check file has columns labeled,date,exdate,cp_flag,best_bid,best_offer")
    
    ###Renaming Columns
    option_df.columns=["date","exdate","best_offer","best_bid","strike","option_right","volume"]

    option_df['qoute_dt'] =pd.to_datetime(option_df['date'])
    option_df['expiry_dt'] =pd.to_datetime(option_df['exdate'])
    option_df = option_df.merge(stock_df, how='left', left_on='qoute_dt', right_on='Date_column')
    option_df = option_df.rename(columns={'price': 'stock_price'})
    option_df = option_df.sort_values(by=['qoute_dt', 'expiry_dt', 'strike'], ascending=[True, True, True])

    ###Strike
    option_df["strike"]=option_df["strike"]/1000
    #### Converting "calls" and "puts" to "c" and "p" respectively.
    option_df['option_right'] = option_df['option_right'].replace({'C': 'c', 'P': 'p'})
    if option_right=="calls":
        option_df=option_df[option_df["option_right"]=="c"]
    if option_right=="puts":
        option_df=option_df[option_df["option_right"]=="p"]
    ###Creating maturity in year fraction. For using black scholes formula I replace 0 with 0.001 for numerical reasons.
    option_df['maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365
    option_df['maturity'] =option_df['maturity'].replace(0, 0.001)
    option_df['maturity']=option_df['maturity']+(15/24)/365 #expires 10pm, qouted 2am.
    option_df['rounded_maturity']= ((pd.to_datetime(option_df['exdate']) - pd.to_datetime(option_df['date'])).dt.days)/365

    ###Creating mid price. Removing any instance in which the mid price is negative
    option_df['mid_price']=(option_df['best_bid']+option_df['best_offer'])/2
    option_df=option_df[option_df['mid_price']>0]

    ##Creating Log-moneyness column
    option_df.loc[option_df['option_right']=='p','log moneyness'] = np.log(option_df['strike']/option_df['stock_price'])
    option_df.loc[option_df['option_right']=='c','log moneyness'] = np.log(option_df['stock_price']/option_df['strike'])
    #option_df=option_df[option_df["mid_price"]>0.5]
    ###
    
    
    option_df=pasting_yields_option_df(option_df, spot_curve_df, "risk_free_rate")
    option_df=pasting_yields_option_df(option_df, forward_curve_df, "forward_risk_free_rate")
    ##
    ##merging yield curve onto main option dataframe
    # option_df = option_df.merge(yield_curve_df, how='left', left_on='qoute_dt', right_on='Date')
    # option_df.drop(columns=['Date'], inplace=True)
    # date_group=option_df.groupby(['date'])
    # yield_curve_headers=yield_curve_df.columns.tolist()[1:]
    
    # print("--" * 20)  # Print a line of dashes for separation
    # print("We are interpolating the yield curve to match your options. This may take some time.")
    # ###Performing Yield Curve interpolation
    # option_df=option_df.reset_index(drop=True)
    # option_df=date_group.apply(lambda x: paste_rate(x,yield_curve_headers)) 
    # print("We are done interpolating Yield curve")
    # option_df=option_df.reset_index(drop=True)
    
    print("--" * 20)  # Print a line of dashes for separation
    print("Converting puts to calls. If both are present we use only OTM options.")
    date_group=option_df.groupby(['date','exdate'])
    option_df=date_group.apply(lambda x: use_only_OTM(x))
    option_df=option_df.reset_index(drop=True)
    
    date_group=option_df.groupby(['date','exdate'])
    print("--" * 20)  # Print a line of dashes for separation
    print("We are Aggregating duplicate option entries")    
    option_df=date_group.apply(lambda x: remove_duplicates(x))
    option_df=option_df.reset_index(drop=True)
    option_df['option_right'] = option_df['option_right'].replace({'call parity': 'c'})  
    
    ###Repair arbitrage
    option_df["underlying_price"]=option_df["stock_price"]
    option_df=option_df.reset_index(drop=True)
    # date_group=option_df.groupby(['date','exdate'])
    # option_df["repair_price"]=0
    # option_df=date_group.apply(lambda x: pre_arbitrage_repair(x))
    ##
    
    
    mid_iv=py_vollib.black.implied_volatility.implied_volatility(option_df["mid_price"], 
                                                                 option_df["stock_price"], option_df["strike"], option_df["risk_free_rate"], option_df["maturity"], 'c', return_as='numpy')
    option_df["mid_iv"]=mid_iv
    option_df=option_df.reset_index(drop=True)
    option_df = option_df.dropna()
    option_df['date'] = pd.to_datetime(option_df['date']).dt.strftime('%Y-%m-%d')
    option_df['exdate'] = pd.to_datetime(option_df['exdate']).dt.strftime('%Y-%m-%d')
    
    option_df["bid_size"]=option_df["volume"]
    option_df["offer_size"]=option_df["volume"]
    option_df["moneyness"]=option_df["strike"]/option_df['stock_price']-1
    option_df=option_df[["date","exdate","maturity","rounded_maturity","risk_free_rate","forward_risk_free_rate","strike","best_bid","mid_price","best_offer","mid_iv","underlying_price","stock_price",
                        "volume","offer_size","bid_size","option_right","moneyness","qoute_dt","expiry_dt"]]
    return option_df

def rolling_window(df,nearest_maturity):
    """
    Parameters
    ----------
    df : Date-slice expiration dataframe
        Date_group:
    nearest_maturity : float
        Keeps only options with maturity closest to this number.

    Returns
    -------
    df : dataframe
        contains only options with maturies nearest to the nearest maturity.

    """
    
    df=df.copy()
    unique_maturity_list = df["rounded_maturity"].unique()*365
    closest_maturity = round(unique_maturity_list[np.abs(unique_maturity_list - nearest_maturity).argmin()])/365
    
    df=df[df["rounded_maturity"]==closest_maturity]
    
    return df

def remove_duplicates(df):
    """
    Helper function that removes prices in the option dataframe. Prices should be monotonically decreasing/decreasing with respect to strike. In deribit data sometimes option prices will have the same price
    even as the strike prices increases/decreases. This usually only happens at the end of the strike domain. I truncate the data at the first occurence of the price repeating.

    Parameters
    ----------
    df : Dataframe
        Option dataframe. Should be a Date-exdate slice.

    Returns
    -------
    df : Dataframe
        Original Dataframe bu with mid prices truncated.

    """
    df=df.reset_index(drop=True)
    df = df[~df.duplicated(subset=['mid_price'], keep='first')]
    strike_list=df.loc[:,'strike'].values
    
    unique_values, counts = np.unique(strike_list, return_counts=True)
    has_multiple_duplicates = np.any(counts > 1)

    if has_multiple_duplicates:
        def custom_agg_func(series):
            """
            Helper function that handles DF where there are duplicate strikes by averaging.
    
            Parameters
            ----------
            series : TYPE
                DESCRIPTION.
    
            Returns
            -------
            TYPE
                DESCRIPTION.
    
            """
            if pd.api.types.is_numeric_dtype(series):
                return series.mean()
            else:
                return series.iloc[0]
        
        # Apply the custom aggregation function
        df = df.groupby('strike', as_index=False).agg(custom_agg_func)
    #print('working remove')
    return df


def use_only_OTM(option_df):
    """
    Description of function: Converts puts into calls via put call parity, and replaces puts with calls.

    Parameters
    ----------
    option_df : dataframe
        option dataframe (day-expiration slice).

    Returns
    -------
    option_df : dataframe
        Dataframe with puts replaced with calls.

    """
    option_df=option_df.reset_index(drop=True)

    contains_only_c = all(x == 'c' for x in option_df['option_right'].to_list())
    contains_only_p = all(x == 'p' for x in option_df['option_right'].to_list())

    if contains_only_c==True:
        #print("Keeping all Calls.")
        return option_df
    if contains_only_p==True:
        #print("Converting puts into calls via put call parity")
        condition = (option_df['option_right'] == "p")
        option_df.loc[condition,'mid_price']=option_df["mid_price"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        option_df.loc[condition,'best_bid']=option_df["best_bid"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        option_df.loc[condition,'best_offer']=option_df["best_offer"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        
        option_df.loc[condition,'option_right']='c'
    if ((contains_only_c==False) & (contains_only_p==False)):
        #print("Replacing ITM calls with puts via put call parity")
        condition = (option_df['option_right'] == "p") & (option_df['strike'] < option_df['stock_price']) ##otm puts
        option_df.loc[condition,'mid_price']=option_df["mid_price"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"] ###convert puts into calls
        option_df.loc[condition,'best_bid']=option_df["best_bid"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        option_df.loc[condition,'best_offer']=option_df["best_offer"]-option_df["strike"]*np.exp(-option_df["risk_free_rate"]*option_df["maturity"])+option_df["stock_price"]
        
        option_df.loc[condition,'option_right']='call parity' #otm puts relabedl as call parity
        option_df = option_df[~(option_df['option_right'] == 'p')] #ITM puts removed
        condition=(option_df['option_right']=='c') & (option_df['strike']<option_df['stock_price']) ##ITM calls
        option_df=option_df.loc[~condition] ##ITM calls removed
        option_df.loc[condition,'option_right']='call parity' #otm puts relabedl as call parity

    return option_df
def average_futures(df):
    """
    Helper function that averages the futures prices on a given day, for each maturity. This is a known issue with deribit, where for some reason they quote multiple futures prices for a single maturity.
    I take the mean of these and use it as the underlying.

    Parameters
    ----------
    df : Dataframe date-expiry slice
        Dataframe with date-expiry slice.

    Returns
    -------
    df : Dataframe
        Original dataframe with futures price averaged.

    """
    df=df.reset_index(drop=True)
    mean_underlying_future=np.mean(df['underlying_price'].values)
    df.loc[:,'underlying_price']=mean_underlying_future
    return df


def pasting_yields_option_df(option_df,yields_df,new_column_name):
    """
    Description of function: Pastes interpolated risk free rates from the yields_df onto the options df

    Parameters
    ----------
    option_df : TYPE
        DESCRIPTION.
    yields_df : TYPE
        DESCRIPTION.
    new_column_name : TYPE
        DESCRIPTION.

    Returns
    -------
    option_df : TYPE
        DESCRIPTION.

    """
    # Reshape spot_curves_df to long format
    yields_df_long = yields_df.reset_index().melt(
        id_vars=["index"], var_name="rounded_maturity", value_name=new_column_name
    )
    yields_df_long.rename(columns={"index": "qoute_dt"}, inplace=True)
    
    # Ensure the data types align for merging
    yields_df_long["qoute_dt"] = pd.to_datetime(yields_df_long["qoute_dt"])
    yields_df_long["rounded_maturity"] = yields_df_long["rounded_maturity"].astype(float)/365  # Convert maturity to float for matching
    
    # Assuming 'option_df' is your option dataframe
    option_df["qoute_dt"] = pd.to_datetime(option_df["qoute_dt"])  # Ensure 'date' is datetime
    option_df["maturity"] = option_df["maturity"].astype(float)  # Ensure 'maturity' is float for matching
    
    # Merge the option dataframe with the reshaped spot curves
    option_df = option_df.merge(
        yields_df_long,
        how="left",  # Use 'left' join to keep all rows in the option dataframe
        on=["qoute_dt", "rounded_maturity"]
    )
    return option_df