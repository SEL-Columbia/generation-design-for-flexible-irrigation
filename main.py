from capacity_model import create_capacity_model
from operation_model import create_operation_model
from utils import *
import datetime

if __name__ == '__main__':
    running_start_time = datetime.datetime.now()

    args = get_args()
    for sce_sf_area_m2 in args.sf_land_area:
        sce_sf_area_m2 = float(sce_sf_area_m2)
        scenario_name = 'Region_' + str(args.region_no) + '_Config_' + str(args.config)

        # Phase 1 - no irrigation minimum power, capacity model
        nodes_capacity_results = create_capacity_model(args, sce_sf_area_m2)
        # Phase 2 - with irrigation minimum power, operation model
        create_operation_model(args, nodes_capacity_results, scenario_name, sce_sf_area_m2)

    running_end_time = datetime.datetime.now()
    print(running_end_time - running_start_time)