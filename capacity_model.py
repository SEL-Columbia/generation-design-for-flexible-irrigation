from gurobipy import *
from utils import get_cap_cost, load_timeseries, get_nodes_area
from results_processing import node_results_retrieval
import numpy as np
import pandas as pd

def create_capacity_model(args, sce_sf_area_m2):
    print("capacity model building and solving")
    print("--------####################------------")
    T = args.num_hours
    trange = range(T)
    # config the connection
    num_nodes, irrigation_area_m2 = get_nodes_area(args, sce_sf_area_m2)
    print('num_nodes:', num_nodes,  ' & its areas [m2]:', np.sum(irrigation_area_m2))

    dome_load_hourly_kw, solar_po_hourly, rain_rate_daily_mm_m2 = load_timeseries(args)

    # get the capital price
    solar_cap_cost, battery_la_cap_cost_kwh, battery_li_cap_cost_kwh, battery_inverter_cap_cost_kw, \
    diesel_cap_cost_kw = get_cap_cost(args, args.num_year)

    # initial the results table
    nodes_results = pd.DataFrame()

    # set up regional model
    for i in range(num_nodes):
        m = Model("capacity_model_node_" + str(i))
        print('capacity model node ' + str(i) + ' building and solving')

        # read the domestic load in each region, scaled by the region area
        if args.config == 0:
            dome_load = dome_load_hourly_kw / 100 * args.dome_load_rate * (sce_sf_area_m2 / 10000)
        else:
            dome_load = dome_load_hourly_kw / 100 * args.dome_load_rate * (irrigation_area_m2[i] / 10000)

        # Initialize capacity variables / bind the capacity from the dry season days model
        solar_cap = m.addVar(name='solar_cap')
        solar_binary = m.addVar(name='solar_cap_binary', vtype=GRB.BINARY)
        m.setPWLObj(solar_cap, args.solar_pw_cap_kw, solar_cap_cost)
        diesel_cap = m.addVar(obj=diesel_cap_cost_kw, name='diesel_cap')
        diesel_binary = m.addVar(name='diesel_cap_binary', vtype=GRB.BINARY)
        battery_la_cap_kwh = m.addVar(obj=battery_la_cap_cost_kwh, name = 'batt_la_energy_cap')
        battery_la_cap_kw  = m.addVar(obj=battery_inverter_cap_cost_kw, name = 'batt_la_power_cap')
        battery_li_cap_kwh = m.addVar(obj=battery_li_cap_cost_kwh, name = 'batt_li_energy_cap')
        battery_li_cap_kw  = m.addVar(obj=battery_inverter_cap_cost_kw, name = 'batt_li_power_cap')

        # constraints for tech availability
        if not args.solar_ava:
            m.addConstr(solar_cap == 0)
        else:
            m.addConstr(solar_cap - args.solar_min_cap * solar_binary >= 0)
            m.addConstr(solar_cap * (1-solar_binary) == 0)

        if not args.battery_la_ava:
            m.addConstr(battery_la_cap_kwh == 0)
        if not args.battery_li_ava:
            m.addConstr(battery_li_cap_kwh == 0)

        if not args.diesel_ava:
            m.addConstr(diesel_cap == 0)
        else:
            m.addConstr(diesel_cap - args.diesel_min_cap * diesel_binary >= 0)
            m.addConstr(diesel_cap * (1-diesel_binary) == 0)
            if args.diesel_vali_cond:
                m.addConstr(diesel_binary == 1) # add the post-model validation back to the model

        # battery capacity constraints
        m.addConstr(battery_la_cap_kwh * (1-args.battery_la_min_soc) * float(args.battery_la_p2e_ratio_range[0]) <=
                    battery_la_cap_kw)
        m.addConstr(battery_la_cap_kwh * (1-args.battery_la_min_soc) * float(args.battery_la_p2e_ratio_range[1]) >=
                    battery_la_cap_kw)
        m.addConstr(battery_li_cap_kwh * (1-args.battery_li_min_soc) * float(args.battery_li_p2e_ratio_range[0]) <=
                    battery_li_cap_kw)
        m.addConstr(battery_li_cap_kwh * (1-args.battery_li_min_soc) * float(args.battery_li_p2e_ratio_range[1]) >=
                    battery_li_cap_kw)
        m.update()


        # Initialize time-series variables
        irrigation_load = m.addVars(trange, obj=args.irrigation_nominal_cost, name='irrigation_load')
        solar_util = m.addVars(trange, name='solar_util')

        battery_la_charge = m.addVars(trange, obj=args.nominal_charge_discharge_cost_kwh, name='batt_la_charge')
        battery_la_discharge = m.addVars(trange, obj=args.nominal_charge_discharge_cost_kwh, name='batt_la_discharge')
        battery_la_level = m.addVars(trange, name='batt_la_level')
        battery_li_charge = m.addVars(trange, obj=args.nominal_charge_discharge_cost_kwh, name='batt_li_charge')
        battery_li_discharge = m.addVars(trange, obj=args.nominal_charge_discharge_cost_kwh, name='batt_li_discharge')
        battery_li_level = m.addVars(trange, name='batt_li_level')

        diesel_kwh_fuel_cost = args.diesel_cost_liter * args.liter_per_kwh / args.diesel_eff
        diesel_gen = m.addVars(trange, obj=diesel_kwh_fuel_cost, name="diesel_gen")
        m.update()

        # Add time-series Constraints
        for j in trange:
            # solar and diesel generation constraint
            m.addConstr(diesel_gen[j] <= diesel_cap)
            m.addConstr(solar_util[j] <= solar_cap * solar_po_hourly[j])

            # Energy Balance
            m.addConstr(solar_util[j] + diesel_gen[j] - battery_la_charge[j] + battery_la_discharge[j] -
                        battery_li_charge[j] + battery_li_discharge[j] == dome_load[j] + irrigation_load[j])

            # Battery operation constraints
            m.addConstr(args.battery_la_eff * battery_la_charge[j] - battery_la_cap_kw <= 0)
            m.addConstr(battery_la_discharge[j] / args.battery_la_eff - battery_la_cap_kw <= 0)
            m.addConstr(battery_la_level[j] - battery_la_cap_kwh <= 0)
            m.addConstr(battery_la_level[j] - battery_la_cap_kwh * args.battery_la_min_soc >=0)

            m.addConstr(args.battery_li_eff * battery_li_charge[j] - battery_li_cap_kw <= 0)
            m.addConstr(battery_li_discharge[j] / args.battery_li_eff - battery_li_cap_kw <= 0)
            m.addConstr(battery_li_level[j] - battery_li_cap_kwh <= 0)
            m.addConstr(battery_li_level[j] - battery_li_cap_kwh * args.battery_li_min_soc >=0)

            ## Battery control
            if j == 0:
                m.addConstr(battery_la_discharge[j] / args.battery_la_eff - args.battery_la_eff * battery_la_charge[j] ==
                            battery_la_level[T - 1] - battery_la_level[j])
                m.addConstr(battery_li_discharge[j] / args.battery_li_eff - args.battery_li_eff * battery_li_charge[j] ==
                            battery_li_level[T - 1] - battery_li_level[j])
            else:
                m.addConstr(battery_la_discharge[j] / args.battery_la_eff - args.battery_la_eff * battery_la_charge[j] ==
                            battery_la_level[j - 1] - battery_la_level[j])
                m.addConstr(battery_li_discharge[j] / args.battery_li_eff - args.battery_li_eff * battery_li_charge[j] ==
                            battery_li_level[j - 1] - battery_li_level[j])
        m.update()

        # Irrigation + Rain Rate Constraints:
        #   1. create water storage in soil
        #   2. constrains on irrigation
        day_range = range(int(T/24))
        ground_water_level_mm = m.addVars(day_range, obj=args.nominal_water_level, name='ground_water_level_mm')
        ground_water_charge_mm = m.addVars(day_range, name='ground_water_charge_mm')
        ground_water_discharge_mm = m.addVars(day_range, obj=args.nominal_water_discharge, name='ground_water_discharge_mm')
        m.update()
        if args.config >= 1:
            m.addConstr(ground_water_level_mm[args.first_season_start] == 0)
            m.addConstr(ground_water_level_mm[args.second_season_start] == 0)
            for d in list(range(args.first_season_start, args.first_season_end+1)) + \
                     list(range(args.second_season_start, args.second_season_end+1)):
                irrigation_daily_mm = quicksum(irrigation_load[k] for k in range((d*24), ((d+1)*24))) / \
                                      args.irrigation_kwh_p_kg / irrigation_area_m2[i]
                m.addConstr(rain_rate_daily_mm_m2[d] + irrigation_daily_mm + ground_water_discharge_mm[d] >=
                            args.water_demand_kg_m2_day + ground_water_charge_mm[d])
                m.addConstr(ground_water_level_mm[d+1] == ground_water_level_mm[d] +
                            ground_water_charge_mm[d] - ground_water_discharge_mm[d])
                m.addConstr(ground_water_level_mm[d+1] <= (args.water_account_days-1) * args.water_demand_kg_m2_day)

            for d in list(range(args.first_season_end+1, args.second_season_start)) + \
                     list(range(args.second_season_end+1, 365)):
                irrigation_daily_mm = quicksum(irrigation_load[k] for k in range((d*24), ((d+1)*24))) / \
                                      args.irrigation_kwh_p_kg / irrigation_area_m2[i]
                m.addConstr(irrigation_daily_mm == 0)
                m.addConstr(ground_water_level_mm[d] == 0)
                m.addConstr(ground_water_charge_mm[d] == 0)
                m.addConstr(ground_water_discharge_mm[d] == 0)

        m.update()

        # Set model solver parameters
        m.setParam("FeasibilityTol", args.feasibility_tol)
        m.setParam("OptimalityTol",  args.optimality_tol)
        m.setParam("Method",         args.solver_method)
        # Solve the model
        m.optimize()

        ### ------------------------- Results Output ------------------------- ###
        # Retrieve results and process the model solution for next step
        single_node_results, single_node_ts_results = node_results_retrieval(args, m, i, T, sce_sf_area_m2)
        nodes_results = nodes_results.append(single_node_results)
    nodes_results = nodes_results.reset_index(drop=True)

    return nodes_results
