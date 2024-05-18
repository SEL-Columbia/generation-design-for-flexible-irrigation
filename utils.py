import os, re, argparse, yaml
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def get_args():
    # Store all parameters for easy retrieval
    parser = argparse.ArgumentParser(description = 'fixed&flexible')
    parser.add_argument('--params_filename',
                        type=str,
                        default='params.yaml',
                        help = 'Loads model parameters')
    args = parser.parse_args()
    config = yaml.load(open(args.params_filename), Loader=yaml.FullLoader)
    for k,v in config.items():
        args.__dict__[k] = v
    return args

def get_nodes_area(args, sce_sf_area_m2):
    # base on the config to get the nodes number and area
    if args.config == 0:
        num_nodes = 1
        irrigation_area_m2 = [0]
    elif args.config == 0.5:
        num_nodes, irrigation_area_m2 = read_irrigation_area(args)
        num_nodes = 1
        irrigation_area_m2 = [float(np.sum(irrigation_area_m2))]
    elif args.config == 1:
        num_nodes = 1
        irrigation_area_m2 = [sce_sf_area_m2]
    elif args.config == 2:
        num_nodes, irrigation_area_m2 = read_irrigation_area(args)
    elif args.config == 3 or args.config == 4:
        num_nodes, irrigation_area_m2 = read_irrigation_area(args)
        num_nodes = 1
        irrigation_area_m2 = [float(np.sum(irrigation_area_m2))]
    return num_nodes, irrigation_area_m2

def annualization_rate(i, years):
    return (i*(1+i)**years)/((1+i)**years-1)

def get_cap_cost(args, years):
    # Annualize capacity costs for model
    annualization_solar   = annualization_rate(args.i_rate, args.annualize_years_solar)
    annualization_battery_la = annualization_rate(args.i_rate, args.annualize_years_battery_la)
    annualization_battery_li = annualization_rate(args.i_rate, args.annualize_years_battery_li)
    annualization_battery_inverter = annualization_rate(args.i_rate, args.annualize_years_battery_inverter)
    annualization_diesel  = annualization_rate(args.i_rate, args.annualize_years_diesel)
    # only solar will use piecewise capital cost
    solar_cap_cost = [years * annualization_solar * float(solar_cost) for solar_cost in args.solar_pw_cost_kw]
    battery_la_cap_cost_kwh  = years * annualization_battery_la * float(args.battery_la_cost_kwh)
    battery_li_cap_cost_kwh  = years * annualization_battery_li * float(args.battery_li_cost_kwh)
    battery_inverter_cap_cost_kw  = years * annualization_battery_inverter * float(args.battery_inverter_cost_kw)
    diesel_cap_cost_kw = years * annualization_diesel * float(args.diesel_cap_cost_kw)
    return solar_cap_cost, battery_la_cap_cost_kwh, battery_li_cap_cost_kwh, \
           battery_inverter_cap_cost_kw, diesel_cap_cost_kw

def load_timeseries(args):
    # Load solar & load time series, all region use the same
    solar_po_hourly   = np.array(pd.read_csv(f'{args.data_dir}/region_{str(args.region_no)}/solar_po_2014_2015.csv', index_col=0))[:,0]
    fix_load_hourly_kw = np.array(pd.read_csv(f'{args.data_dir}/fixed_load_kw.csv', index_col=0))[:,0]
    rain_rate_daily_mm_m2 = np.array(pd.read_csv(f'{args.data_dir}/rain_rate_mm_2014_2015.csv', index_col=0))[:,0]
    return fix_load_hourly_kw, solar_po_hourly, rain_rate_daily_mm_m2

def read_irrigation_area(args):
    irrigation_area_m2 = pd.read_csv(os.path.join(args.data_dir, 'region_{}'.format(str(args.region_no)), 'pts_area.csv'))["AreaSqM"] * args.irrgation_area_ratio
    num_regions = len(irrigation_area_m2)
    return num_regions, irrigation_area_m2

def get_connection_info(args):
    # transmission connection is pre-solved with TLND model
    f = open(os.path.join(args.data_dir,'region_{}'.format(str(args.region_no)),'irrig_zones_centroidsmodelOutput.txt'),'r')
    tx_f = f.read()
    lv_length = float(re.search(r'LVLength:(.*?)\n',tx_f).group(1))
    mv_length = float(re.search(r'MVLength:(.*?)\n',tx_f).group(1))
    tx_num = float(re.search(r'Num Transformers:(.*?)\n',tx_f).group(1))
    print("connection info", lv_length, mv_length, tx_num)
    return lv_length, mv_length, tx_num

