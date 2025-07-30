import opentrons
import time
from math import ceil
from opentrons import protocol_api
import random


metadata = {
    "protocolName": "Barcoding 16-96 samples V2.1",
    "description": """Barcoding up to 96 samples in 96 well deepwell block""",
    "author": "Jakob Einhaus"
    }

requirements = {"robotType": "Flex", "apiLevel": "2.19"}

steps = [] 

# Initialize the previous x position
previous_x = random.uniform(-100, 0)
previous_x_eightchannel = previous_x
previous_x_singlechannel = previous_x

def add_parameters(parameters: protocol_api.Parameters):

    parameters.add_str(
    variable_name="tip_destiny",
    display_name="Tip destiny",
    choices=[
        {"display_name": "Trash", "value": "Trash"},
        {"display_name": "Return to rack", "value": "Return"},
    ],
    default="Trash",
    )

    parameters.add_int(
    variable_name="nsamples",
    display_name="Number of Samples",
    description="Number of samples to barcode",
    default=96,
    minimum=16,
    maximum=96,
    unit="samples")
 
    parameters.add_str(
    variable_name="starting_step",
    display_name="Starting Step",
    choices=[
        {"display_name": "PBS wash", "value": "PBS_wash"},
        {"display_name": "Fill saponin block", "value": "distribute_saponin"},
        {"display_name": "Saponin/PBS wash", "value": "saponin_wash"},
        {"display_name": "Combine barcode/saponin/PBS", "value": "barcode_into_saponin"},
        {"display_name": "Barcode cells", "value": "barcode_and_saponin_into_cells"},
        {"display_name": "First CSM wash", "value": "1st_CSM_wash"},
        {"display_name": "Second CSM wash", "value": "2nd_CSM_wash"},
        {"display_name": "Third CSM wash", "value": "3rd_CSM_wash"},
        {"display_name": "Pool wells together", "value": "1st_pooling_step"},
        {"display_name": "Wash wells with CSM", "value": "wash_wells_CSM"},
        {"display_name": "Pool CSM wash together", "value": "2nd_pooling_step"},
    ],
    default="PBS_wash",
    )

    parameters.add_str(
    variable_name="stopping_step",
    display_name="Stopping Step",
    choices=[
        {"display_name": "PBS wash", "value": "PBS_wash"},
        {"display_name": "Fill saponin block", "value": "distribute_saponin"},
        {"display_name": "Saponin/PBS wash", "value": "saponin_wash"},
        {"display_name": "Combine barcode/saponin/PBS", "value": "barcode_into_saponin"},
        {"display_name": "Barcode cells", "value": "barcode_and_saponin_into_cells"},
        {"display_name": "First CSM wash", "value": "1st_CSM_wash"},
        {"display_name": "Second CSM wash", "value": "2nd_CSM_wash"},
        {"display_name": "Third CSM wash", "value": "3rd_CSM_wash"},
        {"display_name": "Pool wells together", "value": "1st_pooling_step"},
        {"display_name": "Wash wells with CSM", "value": "wash_wells_CSM"},
        {"display_name": "Pool CSM wash together", "value": "2nd_pooling_step"},
    ],
    default="2nd_pooling_step",
    )


def run(protocol: protocol_api.ProtocolContext):
    # A1
    tips50 = protocol.load_labware("opentrons_flex_96_tiprack_50ul", "A1")
    # A2
    barcodeplate = protocol.load_labware("pcr_barcodeplate", "A2")
    # A3
    trash = protocol.load_trash_bin("A3")
    # B1
    
    # B2
    saponinblock = protocol.load_labware("vwr_96_wellplate_2500ul", "B2")   
    # B3
    tips1000A = protocol.load_labware("opentrons_flex_96_tiprack_1000ul", "B3")
    # C1
    falcons = protocol.load_labware("opentrons_15_tuberack_falcon_15ml_conical", "C1")
    # C2
    cellblock = protocol.load_labware("vwr_96_wellplate_2500ul", "C2")
    # C3
    saponin_reservoir = protocol.load_labware("vwr_290ml_reservoir", "C3")
    # D1
    tips1000B = protocol.load_labware("opentrons_flex_96_tiprack_1000ul", "D1")
    # D2
    csm_reservoir = protocol.load_labware("vwr_290ml_reservoir", "D2")
    # D3
    pbs_reservoir = protocol.load_labware("vwr_290ml_reservoir", "D3")

    
    singlechannel = protocol.load_instrument("flex_1channel_1000", "left", tip_racks=[tips1000A, tips1000B, tips50])
    eightchannel = protocol.load_instrument("flex_8channel_1000", "right", tip_racks=[tips1000A, tips1000B, tips50]) 
    
    ncol = ceil(protocol.params.nsamples / 8)
    targets = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12"]
    targets = targets[:ncol]

   

    def get_random_x_singlechannel():
        global previous_x_singlechannel
        while True:
            new_x = random.uniform(-100, 10)
            if abs(new_x - previous_x_singlechannel) > 20:  # Ensure it's at least 40 units away
                previous_x_singlechannel = new_x
                return new_x
    
    def get_random_x_eightchannel():
        global previous_x_eightchannel
        while True:
            new_x = random.uniform(-100, 60)
            if abs(new_x - previous_x_eightchannel) > 40:  # Ensure it's at least 40 units away
                previous_x_eightchannel = new_x
                return new_x

    def wash(vol, tips, res, dest, res_dynamic= True, putback = False): 
        eightchannel.flow_rate.aspirate = 600
        eightchannel.flow_rate.dispense = 600
        eightchannel.flow_rate.blow_out = 500
        eightchannel.well_bottom_clearance.aspirate = 2
        eightchannel.well_bottom_clearance.dispense = 40

        if not eightchannel.has_tip:
            eightchannel.pick_up_tip(tips)

        eightchannel.configure_for_volume(vol+20)
        eightchannel.aspirate(20, location = res["A1"])
        for y in targets:
            eightchannel.dispense(10, location = res[y].bottom(2) if res_dynamic else res["A1"].bottom(2))
            eightchannel.aspirate(vol, location = res[y] if res_dynamic else res["A1"])
            eightchannel.air_gap(10)
            eightchannel.dispense(vol+10, location=dest[y])  
            eightchannel.air_gap(10)      
        eightchannel.dispense(20, location = res["A1"])     

        if putback:
            eightchannel.blow_out(location = res["A1"])
            eightchannel.return_tip()   

    def prepare_barcode(): 
        eightchannel.flow_rate.aspirate = 20
        eightchannel.flow_rate.dispense = 10
        eightchannel.flow_rate.blow_out = 400

        for w in targets: 
            if not eightchannel.has_tip:
                eightchannel.pick_up_tip(tips50)
            eightchannel.configure_for_volume(5)
            eightchannel.well_bottom_clearance.aspirate = 0.1
            eightchannel.aspirate(5, location = barcodeplate[w])
            eightchannel.air_gap(2)
            eightchannel.dispense(5+2, location=saponinblock[w].bottom(1), push_out=10)
            eightchannel.mix(repetitions = 2, location =saponinblock[w].bottom(10), volume=40, rate=3)
            eightchannel.blow_out()    
            if protocol.params.tip_destiny == "Trash":
                eightchannel.drop_tip(location=trash.top(x=get_random_x_eightchannel())) 
            elif protocol.params.tip_destiny == "Return":    
                eightchannel.return_tip() 


    def barcode_cells(): 
        eightchannel.flow_rate.aspirate = 600
        eightchannel.flow_rate.dispense = 600
        eightchannel.flow_rate.blow_out = 400

        for y in targets: 
            eightchannel.pick_up_tip(tips1000A)
            eightchannel.configure_for_volume(500)
            eightchannel.well_bottom_clearance.aspirate = 1
            eightchannel.mix(repetitions = 5, volume=400, location = saponinblock[y], rate = 2)
            eightchannel.aspirate(475, location = saponinblock[y])
            eightchannel.air_gap(10)
            eightchannel.dispense(485, location=cellblock[y], push_out=30)
            eightchannel.mix(repetitions = 5, volume=400, location = cellblock[y], rate = 2)
            eightchannel.blow_out()
            if protocol.params.tip_destiny == "Trash":
                eightchannel.drop_tip(location=trash.top(x=get_random_x_eightchannel())) 
            elif protocol.params.tip_destiny == "Return":    
                eightchannel.return_tip()


    def delay_with_countdown(minutes: int = 0, add_msg: str = '', alt_msg: str = ''):
        total_seconds = minutes * 60
        start_time = time.time()         
        stop_time = start_time + total_seconds
        update_interval = 60  # Update the message every 60 seconds
        last_update_time = time.time()  # Track the last time the message was updated
        delay_time = 0 if protocol.is_simulating() else 1  # Use shorter delay during simulation
        if alt_msg == '':
            protocol.comment(f'Remaining incubation time: {minutes} minutes. {add_msg}')
        else:
            protocol.comment(f'{alt_msg} {add_msg}')

        # Main loop to run the countdown
        for x in range(15000):
            remaining_time = stop_time - time.time()  # Calculate the remaining time in seconds

            # Check if the remaining time is up
            if remaining_time <= 0:
                protocol.comment('Incubation complete.')
                break  # Exit the loop when time is up

            if (time.time() - last_update_time) >= update_interval:
                rem_minutes = ceil(remaining_time / 60)  # Round up remaining time to the next minute
                if alt_msg == '': 
                    protocol.comment(f'Remaining incubation time: {rem_minutes} min. {add_msg}')
                else:
                    protocol.comment(f'{alt_msg} {add_msg}')
                last_update_time = time.time()  # Reset the last update time

            time.sleep(delay_time)   
    
    def addcsm():
        eightchannel.flow_rate.aspirate = 600
        eightchannel.flow_rate.dispense = 600
        eightchannel.flow_rate.blow_out = 500
        eightchannel.well_bottom_clearance.aspirate = 2
        eightchannel.well_bottom_clearance.dispense = 40
        if not eightchannel.has_tip:
            eightchannel.pick_up_tip(tips1000B)

        eightchannel.aspirate((80*len(targets)+30), location = csm_reservoir['A1'])
        eightchannel.air_gap(10)
        for y in targets:
            eightchannel.dispense(80+10, location=cellblock[y])
            eightchannel.air_gap(10)
        eightchannel.dispense(40, location = csm_reservoir['A12'])


    def pool(volume_to_pool, first_pooling=True): 
        eightchannel.flow_rate.aspirate = 100
        eightchannel.flow_rate.dispense = 400
        eightchannel.flow_rate.blow_out = 400
        eightchannel.well_bottom_clearance.dispense = 30
        eightchannel.well_bottom_clearance.blowout = 30
        maxvolume = 800
        eightchannel.configure_for_volume(900)
        
        if first_pooling:
            location = falcons["A5"]
        else:
            location = falcons["B5"]

        if not eightchannel.has_tip:
            eightchannel.pick_up_tip(tips1000B)
        total_volume = 0
        for z in range(min(11, len(targets) - 1), -1, -1):
            eightchannel.well_bottom_clearance.aspirate = 1
            eightchannel.aspirate(volume_to_pool, location = cellblock[targets[z]])
            total_volume = total_volume + volume_to_pool
            if total_volume > maxvolume:
                eightchannel.dispense(total_volume, location = cellblock['A1'])
                total_volume = 0
        eightchannel.dispense(total_volume, location = cellblock['A1'], push_out= 30)
        if protocol.params.tip_destiny == "Trash":
            eightchannel.drop_tip(location=trash.top(x=get_random_x_eightchannel())) 
        elif protocol.params.tip_destiny == "Return":    
            eightchannel.return_tip() 

        if not singlechannel.has_tip:
            singlechannel.pick_up_tip(tips1000B)
        for well in cellblock.columns_by_name()["1"]:
            singlechannel.aspirate(700, location = well.bottom(1))
            singlechannel.dispense(700, location = location.bottom(100))
            singlechannel.aspirate(700, location = well.bottom(1))
            singlechannel.dispense(700, location = location.bottom(100))
        if protocol.params.tip_destiny == "Trash":
            singlechannel.drop_tip(location=trash.top(x=get_random_x_singlechannel())) 
        elif protocol.params.tip_destiny == "Return":    
            singlechannel.return_tip() 

    steps = [
        ("PBS_wash", 
        lambda: protocol.pause(f'''Required volumes: {((len(targets) * 8) + 14)}mL PBS, {((len(targets) * 12) + 16)}mL PBS/Saponin, 
                                and {((len(targets) * 20) + 20)}mL CSM.''') if protocol.params.nsamples != 96 else None, 
        lambda: wash(tips=tips1000A, res=pbs_reservoir, res_dynamic=False, vol=950, dest = cellblock),
        lambda: eightchannel.move_to(trash)), 

        ("distribute_saponin", 
        lambda: protocol.pause(f'Centrifuge deepwell block at 600xg for 5 min at 4°C. '
                                'During the centrifugation, proceed the next step to add PBS/Saponin to the saponinblock. ' 
                                'Press CONFIRM now.'),
        lambda: wash(tips=tips1000A, res=saponin_reservoir, res_dynamic=False, vol=495, dest=saponinblock)),

        ("saponin_wash", 
        lambda: protocol.pause('''Confirm that you: 1) Aspirated supernatant, 2) Vortexed to loosen cell pellet, 
                               3) Returned the cell block to location B2.'''),
        lambda: wash(tips=tips1000A, res=saponin_reservoir, res_dynamic=False,vol=950, dest=cellblock, putback=True)),

        ("barcode_into_saponin", 
        lambda: protocol.pause(f'''Centrifuge deepwell block at 600xg for 5 min at 4°C. 
                               During the centrigation, proceed the next step to add the barcode reagent to the saponinblock. 
                               Press CONFIRM now.'''),
        lambda: eightchannel.reset_tipracks(),
        lambda: prepare_barcode()),

        ("barcode_and_saponin_into_cells", 
        lambda: protocol.pause(f'''Aspirate supernatant, vortex to loosen cell pellet, and return the deepwell block to location B2. 
                               When completed, press CONFIRM.'''), 
        lambda: barcode_cells(), 
        lambda: delay_with_countdown(15, add_msg=f'Fill reservoir with {((len(targets) * 20) + 20)}mL CSM.')),

        ("1st_CSM_wash", 
        lambda: wash(tips=tips1000B, res=csm_reservoir, vol=500, dest=cellblock),
        lambda: eightchannel.move_to(csm_reservoir['A1'].top())),

        ("2nd_CSM_wash", 
        lambda: protocol.pause(f'Centrifuge at 600xg for 5 min at 4°C, aspirate supernatant and vortex deepwell block to loosen cell pellet. When completed, press CONFIRM.'),
        lambda: wash(tips=tips1000B, res=csm_reservoir, vol=950, dest=cellblock),
        lambda: eightchannel.move_to(csm_reservoir['A1'].top())),

        ("3rd_CSM_wash", 
        lambda: protocol.pause(f'Centrifuge at 600xg for 5 min at 4°C, aspirate supernatant and vortex deepwell block to loosen cell pellet. When completed, press CONFIRM.'),
        lambda: wash(tips=tips1000B, res=csm_reservoir, vol=950, dest=cellblock),
        lambda: eightchannel.move_to(csm_reservoir['A1'].top())),

        ("1st_pooling_step", 
        lambda: protocol.pause(f'Centrifuge at 600xg for 5 min at 4°C, aspirate supernatant and vortex deepwell block to loosen cell pellet. When completed, press CONFIRM.'),
        lambda: pool(130, first_pooling=True)),

        ("wash_wells_CSM",  
        lambda: addcsm()),

        ("2nd_pooling_step",
        lambda: pool(90, first_pooling=False)),
    ]

    start_index = next(i for i, step in enumerate(steps) if step[0] == protocol.params.starting_step)
    stop_index = next(i for i, step in enumerate(steps) if step[0] == protocol.params.stopping_step)

    for step in steps[start_index:stop_index + 1]:
        for func in step[1:]:
            func()
    
    if eightchannel.has_tip:
        if protocol.params.tip_destiny == "Trash":
            if eightchannel.has_tip:  
                eightchannel.drop_tip(location=trash.top(x=get_random_x_eightchannel())) 
        elif protocol.params.tip_destiny == "Return":  
            if eightchannel.has_tip:    
                eightchannel.return_tip() 
