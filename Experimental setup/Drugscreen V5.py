import opentrons
from opentrons import protocol_api
from math import ceil
import time
from collections import OrderedDict


metadata = {
    "protocolName": "Drugscreen protocol V5",
    "description": """Drug drugprep, preincubation, stimulation, fixation, and lysis.""",
    "author": "Jakob Einhaus"
    }

requirements = {"robotType": "Flex", "apiLevel": "2.19"}

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
    
    parameters.add_str(
    variable_name="starting_step",
    display_name="Starting Step",
    choices=[
        {"display_name": "Drug preparation", "value": "Drugprep"},
        {"display_name": "Adding blood", "value": "Blood"},
        {"display_name": "Stim preparation/addition", "value": "Stims"},
        {"display_name": "Fixation", "value": "Fix"},
        {"display_name": "First round of lysis", "value": "1st_lysis"},
        {"display_name": "Second round of lysis", "value": "2nd_lysis"},
        {"display_name": "CSM wash", "value": "CSM_wash"},
    ],
    default="Drugprep",
    )

    parameters.add_str(
    variable_name="stopping_step",
    display_name="Stopping Step",
    choices=[
        {"display_name": "Drug preparation", "value": "Drugprep"},
        {"display_name": "Adding blood", "value": "Blood"},
        {"display_name": "Stim preparation/addition", "value": "Stims"},
        {"display_name": "Fixation", "value": "Fix"},
        {"display_name": "First round of lysis", "value": "1st_lysis"},
        {"display_name": "Second round of lysis", "value": "2nd_lysis"},
        {"display_name": "CSM wash", "value": "CSM_wash"},
    ],
    default="CSM_wash",
    )


def run(protocol: protocol_api.ProtocolContext):

    # A1
    lysisreservoir = protocol.load_labware("vwr_290ml_reservoir", "A1")
    # A2
    tips1000A = protocol.load_labware("opentrons_flex_96_tiprack_1000ul", "A2")
    # A3
    trash = protocol.load_trash_bin("A3") 
    # B1
    hs_mod = protocol.load_module(module_name="heaterShakerModuleV1", location="B1")
    hs_adapter = hs_mod.load_adapter("opentrons_96_deep_well_adapter")
    cellblock = hs_adapter.load_labware("vwr_96_wellplate_2500ul")
    # B2
    stocks = protocol.load_labware("custom_tuberack", "B2")
    # B3
    tips200 = protocol.load_labware("opentrons_flex_96_tiprack_200ul", "B3")
    # C1
    CSMreservoir = protocol.load_labware("vwr_290ml_reservoir", "C1")
    # C2
    tips1000B = protocol.load_labware("opentrons_flex_96_tiprack_1000ul", "C2")
    # C3
    
    # D1
    reservoir = protocol.load_labware("pipette_basin_holder", "D1")
    # D2
    tips1000C = protocol.load_labware("opentrons_flex_96_tiprack_1000ul", "D2")
    # D3
    
    
    singlechannel = protocol.load_instrument("flex_1channel_1000", "left", tip_racks=[tips1000A])
    eightchannel = protocol.load_instrument("flex_8channel_1000", "right", tip_racks=[tips1000A, tips1000B, tips1000C, tips200]) 

    druglocations = ["B1", "C1", "D1", "E1", "B2", "C2", "D2", "E2", "B3", "C3", "D3", "E3"]

    # FUNCTIONS:
    def prepdrugs():
        singlechannel.configure_for_volume(990)
        singlechannel.flow_rate.aspirate = 700
        singlechannel.flow_rate.dispense = 700
        singlechannel.flow_rate.blow_out = 700
        singlechannel.well_bottom_clearance.aspirate = 10
        singlechannel.well_bottom_clearance.dispense = 40
        height = 86
        
        # Loop through each well in the first row of the drug plate 
        for i, well in enumerate(druglocations):
            singlechannel.pick_up_tip(tips1000B)
            singlechannel.well_bottom_clearance.aspirate = height - 6
            singlechannel.aspirate(990, location=stocks["F1"])
            singlechannel.dispense(990, location=stocks[well])
            singlechannel.mix(repetitions = 6, volume = 800, location=stocks[well].bottom(3), rate = 3)
            height = singlechannel.well_bottom_clearance.aspirate
            
            singlechannel.aspirate(930, location=stocks[well].bottom(1.5))
            singlechannel.dispense(80, location=stocks[well].bottom(30))
            singlechannel.air_gap(5)

            for j in range(min(8, len(cellblock.rows()))):  # Assuming an 8-well column
                blood_well = cellblock.rows()[j][i]  # Move down the column in the blood plate
                singlechannel.dispense(105, location=blood_well.bottom(30))
                singlechannel.air_gap(5)
            
            # Handle tip destiny based on protocol parameters
            if protocol.params.tip_destiny == "Trash":
                singlechannel.drop_tip() 
            elif protocol.params.tip_destiny == "Return":    
                singlechannel.return_tip()


    def addblood():
        eightchannel.configure_for_volume(150)
        eightchannel.flow_rate.aspirate = 400
        eightchannel.flow_rate.dispense = 200
        eightchannel.flow_rate.blow_out = 200
        eightchannel.well_bottom_clearance.aspirate = 0.5
        eightchannel.well_bottom_clearance.dispense = 10
        eightchannel.default_speed = 100 
        vol = 0
        # Iterate over two halves in the row
        for column in cellblock.rows_by_name()["A"]:
            if vol < 100:
                if eightchannel.has_tip:
                    if protocol.params.tip_destiny == "Trash":
                        eightchannel.drop_tip() 
                    elif protocol.params.tip_destiny == "Return":    
                        eightchannel.return_tip()
                eightchannel.pick_up_tip(tips1000B)
                eightchannel.aspirate(620, location = reservoir["A1"])
                eightchannel.air_gap(50)
                vol = 620
            eightchannel.dispense(150, location=column)
            eightchannel.flow_rate.aspirate = 50
            eightchannel.air_gap(50)
            vol -= 100 
        if eightchannel.has_tip:
                    if protocol.params.tip_destiny == "Trash":
                        eightchannel.drop_tip() 
                    elif protocol.params.tip_destiny == "Return":    
                        eightchannel.return_tip()
        eightchannel.default_speed = 300 


    def delay_with_countdown(minutes: int = 0, seconds: int = 0, add_msg: str = '', alt_msg: str = ''):
        total_seconds = minutes * 60 + seconds
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

    def prepstims():
        singlechannel.flow_rate.aspirate = 100
        singlechannel.flow_rate.dispense = 100
        singlechannel.well_bottom_clearance.aspirate = 15
        singlechannel.well_bottom_clearance.dispense = 2
        PBSvolumes = [150, 135, 135, 135, 135, 135, 105, 135]
        stimwells = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
        for v, well, t in zip(PBSvolumes, stimwells, tips200.columns_by_name()["1"]):
            singlechannel.configure_for_volume(v)
            singlechannel.pick_up_tip(t)
            singlechannel.aspirate(v, location=stocks["F1"])
            singlechannel.dispense(v, location = stocks[well])
            if v != 200:
                singlechannel.mix(volume=150, repetitions=5, rate=3)
            singlechannel.return_tip()
    

    def addstim():
        eightchannel.configure_for_volume(5)
        eightchannel.flow_rate.aspirate = 20
        eightchannel.flow_rate.dispense = 20
        eightchannel.well_bottom_clearance.aspirate = 1
        eightchannel.well_bottom_clearance.dispense = 2

        for column in cellblock.rows_by_name()["A"]:
            eightchannel.pick_up_tip(tips200)
            eightchannel.aspirate(10, location = stocks["A1"])
            eightchannel.air_gap(5)
            eightchannel.dispense(15, location = column.bottom(1))
            eightchannel.mix(volume= 150, repetitions=3, rate=10)
            if protocol.params.tip_destiny == "Trash":
                eightchannel.drop_tip() 
            elif protocol.params.tip_destiny == "Return":    
                eightchannel.return_tip()


    def fix():
        hs_mod.deactivate_shaker()
        eightchannel.configure_for_volume(280)
        eightchannel.flow_rate.aspirate = 700
        eightchannel.flow_rate.dispense = 700
        eightchannel.flow_rate.blow_out = 700
        eightchannel.well_bottom_clearance.aspirate = 2
        eightchannel.well_bottom_clearance.dispense = 2

        for column in cellblock.rows_by_name()["A"]:
            eightchannel.pick_up_tip(tips1000C)
            eightchannel.aspirate(280, location = reservoir["A2"])
            eightchannel.dispense(280, location = column)
            eightchannel.mix(repetitions= 3, volume = 500, location=column, rate = 2)
            eightchannel.blow_out()

            if protocol.params.tip_destiny == "Trash":
                eightchannel.drop_tip() 
            elif protocol.params.tip_destiny == "Return":    
                eightchannel.return_tip()
        

    def wash(vol, tips, res, mix = False, order_reversed = False): 
        eightchannel.flow_rate.aspirate = 500
        eightchannel.flow_rate.dispense = 500
        eightchannel.flow_rate.blow_out = 700
        eightchannel.well_bottom_clearance.aspirate = 4
        eightchannel.well_bottom_clearance.dispense = 35

        targets = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']
        if order_reversed:
            targets.reverse()

        half_vol = vol / 2 if vol >= 1000 else None
        if not mix:
            eightchannel.pick_up_tip(tips)
            if half_vol:
                eightchannel.configure_for_volume(half_vol+10)
            else:
                eightchannel.configure_for_volume(vol+10)
            eightchannel.aspirate(10, location=res['A1'])
        for t in targets:
            origin, column = res[t], cellblock[t]
            if mix:
                eightchannel.pick_up_tip(tips)
            if half_vol:
                # Handle half volume twice
                for _ in range(2):
                    eightchannel.aspirate(half_vol, location=origin)
                    eightchannel.dispense(half_vol, location=column)
            else:
                # Handle full volume
                eightchannel.aspirate(vol, location=origin)
                eightchannel.dispense(vol, location=column)
            if mix:
                eightchannel.mix(repetitions= 3, volume = 500, location=column.bottom(5), rate = 2)
            
            if mix:
                if protocol.params.tip_destiny == "Trash":
                    eightchannel.drop_tip() 
                elif protocol.params.tip_destiny == "Return":    
                    eightchannel.return_tip() 
        if not mix:
            if protocol.params.tip_destiny == "Trash":
                eightchannel.drop_tip() 
            elif protocol.params.tip_destiny == "Return":    
                eightchannel.return_tip() 

    steps = [
        ("Drugprep",
        lambda: hs_mod.set_target_temperature(37),
        lambda: protocol.pause('''Confirm that setup is complete (except blood).'''),
        lambda: prepdrugs()), 

        ("Blood", 
        lambda: hs_mod.set_target_temperature(37),
        lambda: protocol.pause('''Confirm that you added 12mL blood in the left pipette basin.'''),
        lambda: addblood(),        
        lambda: hs_mod.open_labware_latch(),
        lambda: protocol.pause('''Incubate blood for 50 minutes in the tissue culture incubator (on shaker).''')),
        
        ("Stims",
        lambda: hs_mod.set_target_temperature(37),
        lambda: protocol.pause('''Confirm that the blood is back and the stims are added.'''),
        lambda: prepstims(),
        lambda: tips200.reset(),
        lambda: hs_mod.close_labware_latch(),
        lambda: addstim(),
        lambda: hs_mod.open_labware_latch(),
        lambda: hs_mod.deactivate_heater(),
        lambda: protocol.pause('''Incubate for 2h in the tissue culture incubator (on shaker).
                                  Confirm after 2h incubation is complete.''')),

        ("Fix",
        lambda: hs_mod.deactivate_heater(),
        lambda: hs_mod.close_labware_latch(),
        lambda: fix(),
        lambda: delay_with_countdown(minutes = 15)),

        ("1st_lysis", 
        lambda: wash(vol=950, tips = tips1000A, res=lysisreservoir, mix=True), 
        lambda: delay_with_countdown(5),
        lambda: hs_mod.open_labware_latch(),
        lambda: protocol.pause('''Confirm that you:
                               > Centrifuged at 600xg for 5 min at RT.
                               > Aspirated supernatant. 
                               > Vortexed to loosen cell pellet.
                               > Returned the cellblock to B1.''')), 

        ("2nd_lysis", 
        lambda: hs_mod.open_labware_latch() if protocol.params.starting_step != '2nd_lysis' else None,
        lambda: protocol.pause('''Confirm that you:
                               > Centrifuged at 600xg for 5 min at RT.
                               > Aspirated supernatant. 
                               > Vortexed to loosen cell pellet.
                               > Returned the cellblock to B1.''') if protocol.params.starting_step != '2nd_lysis' else None,
        lambda: hs_mod.close_labware_latch(),
        lambda: wash(vol=1500, tips=tips1000B, res=lysisreservoir, order_reversed=True), 
        lambda: delay_with_countdown(5),
        lambda: hs_mod.open_labware_latch(),
        lambda: protocol.pause('''Confirm that you:
                               > Centrifuged at 600xg for 5 min at RT.
                               > Aspirated supernatant. 
                               > Vortexed to loosen cell pellet.
                               > Returned the cellblock to B1.''')),
 
        ("CSM_wash",
        lambda: hs_mod.open_labware_latch() if protocol.params.starting_step != 'CSM_wash' else None,
        lambda: protocol.pause('''Confirm that you:
                               > Centrifuged at 600xg for 5 min at RT.
                               > Aspirated supernatant. 
                               > Vortexed to loosen cell pellet.
                               > Returned the cellblock to B1.''') if protocol.params.starting_step != 'CSM_wash' else None,
        lambda: protocol.pause('''Confirm that 180mL CSM are in reservoir C1.'''),
        lambda: hs_mod.close_labware_latch(),
        lambda: wash(vol=1500, tips=tips1000B, res=CSMreservoir),
        lambda: hs_mod.open_labware_latch(),
        lambda: protocol.pause('''Confirm that you:
                               > Centrifuged at 600xg for 5 min at RT.
                               > Aspirated supernatant. 
                               > Vortexed to loosen cell pellet.
                               > Put cell block in -80C freezer.''')),
    ]

    start_index = next(i for i, step in enumerate(steps) if step[0] == protocol.params.starting_step)
    stop_index = next(i for i, step in enumerate(steps) if step[0] == protocol.params.stopping_step)

    hs_mod.open_labware_latch()
    protocol.pause('Put the deepwell block in the heater-shaker module.')
    hs_mod.close_labware_latch()

    for step in steps[start_index:stop_index + 1]:
        for func in step[1:]:
            func()

    hs_mod.deactivate_heater()
    hs_mod.open_labware_latch()

    


    

    







    





