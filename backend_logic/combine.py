
from backend_logic.utils import _energy, get_indices_from_dict, _to_numpy, quadro_quadrant_from_azimuth, seven_one_quadrant_from_azimuth, softmax_with_temp, within_margin, azimuth_from_three_energy
import numpy as np

# two setups for anything non stereo:
# cues: use azimuth without events, so with skip model for only direction cues
# full :use full model and combine logic for event+direction

import json
import os
with open(os.path.join(os.path.dirname(__file__), '..', "data" , 'config.json'), 'r') as f:
    config = json.load(f)

SETUP = config.get("format", "cues")


def combine_stereo(left_events, right_events, left_channel, right_channel):
    return combine_stereo_azimuth(left_events, right_events, left_channel, right_channel)

def combine_quadro(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel):
    if SETUP == "cues":
        return combine_quadro_azimuth(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel)
    else:
        return combine_quadro_confidence_weighted_avg(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel)

def combine_7one(
        fl_events, fr_events, rl_events, rr_events, sl_events, sr_events,
        fl_channel, fr_channel, rl_channel, rr_channel, sl_channel, sr_channel
    ):
    if SETUP == "cues":
        # for 7.1 cues, use quadro azimuth on FL,FR,RL,RR and ignore others for direction
        return combine_7one_azimuth(fl_events, fr_events, rl_events, rr_events, sl_events, sr_events,
                   fl_channel, fr_channel, rl_channel, rr_channel, sl_channel, sr_channel) 
    else:
        # for full 7.1, use combine logic on all channels
        return combine_7one_azimuth_logic(fl_events, fr_events, rl_events, rr_events, sl_events, sr_events,
                   fl_channel, fr_channel, rl_channel, rr_channel, sl_channel, sr_channel)


# funcs that could/are used to combine events from multiple channels:

# ------- STEREO-------

def combine_stereo_logic(left_events, right_events, left_channel, right_channel):
    # Convert torch tensors to numpy on CPU
    left_channel = _to_numpy(left_channel)
    right_channel = _to_numpy(right_channel)

    if left_events["event"] == right_events["event"]:
        left_energy = np.sum(np.square(left_channel))
        right_energy = np.sum(np.square(right_channel))

        final = left_events if left_energy >= right_energy else right_events
        return final
    else:
        return left_events + right_events


def combine_stereo_azimuth(left_events, right_events, left_channel, right_channel):
    """
    Returns stereo azimuth in **degrees**, ranged 0°–360°.
    0° = front, 90° = right, 180° = back, 270° = left.
    """
    left_channel = _to_numpy(left_channel)
    right_channel = _to_numpy(right_channel)


    L = _energy(left_channel)
    R = _energy(right_channel)

    # full azimuth for stereo case is underdetermined
    x = R - L

    if x == 0: 
        theta_deg = 0.0  # middle, so no need to notify
        return []

    elif x > 0:
        theta_deg = 90.0  # right
        right_events["degree"] = theta_deg
        return right_events
        
    else:
        theta_deg = 270.0  # left
        left_events["degree"] = theta_deg
        return left_events

#------- QUADRO -------

def combine_quadro_logic(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel):
    twins_fr = {0: 1, 1: 0, 2: 3, 3: 2}
    idx_to_fr = {0: "GF", 1: "GF", 2: "GR", 3: "GR"}
    twins_lr = {0: 2, 1: 3, 2: 0, 3: 1}
    idx_to_lr = {0: "L", 1: "R", 2: "L", 3: "R"}
    visited = set()
    final = []
    combined = fl_events + fr_events + rl_events + rr_events
    
    

    fl_channel = _to_numpy(fl_channel)
    fr_channel = _to_numpy(fr_channel)
    rl_channel = _to_numpy(rl_channel)
    rr_channel = _to_numpy(rr_channel)
    
    combined_vals = [fl_channel, fr_channel, rl_channel, rr_channel]
    for i in range(len(combined)):
        if i not in visited:
            curr = combined[i]["event"]
            indices = get_indices_from_dict(curr,combined)
            visited.update(indices) # since these are all dupe (might be just 1), visited

            combine_lr = False
            combine_fr = True

            # TODO  if all same drop it
            if len(indices) == 4:
                print("All channels have the same event, dropping all")
                return [] # all same, drop all

            if len(indices) > 1:
                max_energy = -1
                chosen = None
                for idx in indices:
                    energy = _energy(combined_vals[idx]) 
                    if energy > max_energy:
                        max_energy = energy
                        chosen = idx
                
                twin_lr = twins_lr[chosen]
                twin_fr = twins_fr[chosen]

                chosen_energy = _energy(combined_vals[chosen])
                twin_lr_energy = 0
                twin_fr_energy = 0
                for idx in indices:
                    if idx != chosen:
                        if idx == twin_lr:   
                            twin_lr_energy = _energy(combined_vals[twin_lr])
                            if (chosen_energy * 0.85) <= twin_lr_energy: # adapt based on percent verschil
                                combine_lr = True

                        if idx == twin_fr:
                            twin_fr_energy = _energy(combined_vals[twin_fr])
                            if (chosen_energy * 0.85) <= twin_fr_energy: # adapt based on percent verschil
                                combine_fr = True
                
                #print("energy summaries:")
                #print("chosen",combined[chosen]["event"])
                #print(f"chosen_energy: {chosen_energy}, twin_lr_energy: {twin_lr_energy}, twin_fr_energy: {twin_fr_energy}")
                if (combine_lr):
                    if not combine_fr or ( combine_fr and (twin_fr_energy <= twin_lr_energy) ):
                        combined[chosen]["orientation"] = idx_to_lr[chosen]
                        final.append([combined[chosen]])
                elif(combine_fr):
                    combined[chosen]["orientation"] = idx_to_fr[chosen]
                    final.append(([combined[chosen],combined[twin_fr]]))
                else:
                    final.append(([combined[chosen]]))
            else:
                final.append(combined[i])

    return final

# what we can do: all 4-> drop, else for each event argmax and combine using azimuth

# ^^ initial try, following is more advanced logic: using azimuth 


def combine_quadro_intensity(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel):

    idx_to_degree = {0: 330, 1: 30, 2: 240, 3: 120}  # FL, FR, RL, RR
    
    fl_channel = _to_numpy(fl_channel)
    fr_channel = _to_numpy(fr_channel)
    rl_channel = _to_numpy(rl_channel)
    rr_channel = _to_numpy(rr_channel)
    
    # Get energies
    FL = _energy(fl_channel)
    FR = _energy(fr_channel)
    RL = _energy(rl_channel)
    RR = _energy(rr_channel)
    
    energies = [FL, FR, RL, RR]
    total_energy = sum(energies) + 1e-8
    
    # If no energy at all, return empty
    if total_energy < 1e-6:
        return []
    
    # Compute intensity-weighted azimuth
    y = (FR + FL) - (RL + RR)  # front vs back
    x = (FR + RR) - (FL + RL)  # right vs left
    
    theta = np.arctan2(y, x)
    theta_deg = np.degrees(theta)
    
    if theta_deg < 0:
        theta_deg += 360.0
    
    theta_deg = (theta_deg + 90.0) % 360.0
    
    # Get events and their confidences
    combined = [fl_events[0], fr_events[0], rl_events[0], rr_events[0]]
    confidences = [e.get("confidence", 0.0) for e in combined]
    
    # Use highest confidence event
    chosen_idx = int(np.argmax(confidences))
    result = combined[chosen_idx].copy()
    result["degree"] = float(theta_deg)
    
    return result


def combine_quadro_confidence_weighted_avg(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel):
    """
    Returns multiple events, each with their own confidence-weighted degree.
    """
    idx_to_degree = {0: 330, 1: 30, 2: 240, 3: 120}  # FL, FR, RL, RR
    
    # Get all events from each channel (each is a list)
    all_channel_events = [fl_events, fr_events, rl_events, rr_events]
    
    # Collect all unique event types across all channels
    unique_events = {}  # event_name -> list of (channel_idx, event_dict)
    
    for ch_idx, events in enumerate(all_channel_events):
        for event in events:
            event_name = event.get("event", "silence")
            if event_name not in unique_events:
                unique_events[event_name] = []
            unique_events[event_name].append((ch_idx, event))
    
    results = []
    
    for event_name, occurrences in unique_events.items():
        if event_name == "silence":
            continue
        
        # Get confidence per channel for this event (0 if not present)
        confidences = [0.0, 0.0, 0.0, 0.0]
        best_event = None
        best_conf = -1
        
        for ch_idx, event in occurrences:
            conf = event.get("confidence", 0.0)
            confidences[ch_idx] = conf
            if conf > best_conf:
                best_conf = conf
                best_event = event
        
        total_conf = sum(confidences)
        if total_conf < 0.01:
            continue
        
        # Convert degrees to vectors for proper circular averaging
        x_sum = 0.0
        y_sum = 0.0
        for i in range(4):
            rad = np.radians(idx_to_degree[i])
            x_sum += confidences[i] * np.cos(rad)
            y_sum += confidences[i] * np.sin(rad)
        
        # Compute weighted average angle
        weighted_deg = np.degrees(np.arctan2(y_sum, x_sum))
        if weighted_deg < 0:
            weighted_deg += 360.0
        
        result = best_event.copy()
        result["degree"] = float(weighted_deg)
        results.append(result)
    
    if not results:
        return []
    
    # Sort by confidence descending
    results.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
    
    return results


def combine_quadro_confidence(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel):
    """
    Confidence-based quadro combine. better for PRETRAINEDSED, instead of yamnet
    """
    idx_to_degree = {0: 330, 1: 30, 2: 240, 3: 120}  # FL, FR, RL, RR
    
    combined = [fl_events[0], fr_events[0], rl_events[0], rr_events[0]]
    
    # Get confidences for each channel
    confidences = [e.get("confidence", 0.0) for e in combined]
    FL_conf, FR_conf, RL_conf, RR_conf = confidences
    
    # Check if all confidences are too low (silence)
    if max(confidences) < 0.01:
        return []
    
    # Check if confidences are within margin (no dominant direction)
    # But don't drop - use confidence-weighted azimuth
    if within_margin(confidences, margin=0.15):
        # All similar confidence - compute weighted average direction
        total_conf = sum(confidences) + 1e-8
        weighted_degree = sum(idx_to_degree[i] * confidences[i] for i in range(4)) / total_conf
        weighted_degree = weighted_degree % 360.0
        
        # Use highest confidence event
        chosen_idx = np.argmax(confidences)
        result = combined[chosen_idx].copy()
        result["degree"] = float(weighted_degree)
        return result
    
    # Compute azimuth using confidence as pseudo-energy
    # Same formula as energy-based but with confidence
    y = (FR_conf + FL_conf) - (RL_conf + RR_conf)  # front vs back
    x = (FR_conf + RR_conf) - (FL_conf + RL_conf)  # right vs left
    
    theta = np.arctan2(y, x)
    theta_deg = np.degrees(theta)
    
    if theta_deg < 0:
        theta_deg += 360.0
    
    theta_deg = (theta_deg + 90.0) % 360.0
    
    # Get the quadrant with highest confidence
    _, quadrant_idx = quadro_quadrant_from_azimuth(theta_deg)
    
    result = combined[quadrant_idx].copy()
    result["degree"] = float(theta_deg)
    
    return result


def combine_quadro_weighted_confidence(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel):

    to_side_lr = {0: 0, 1: 1, 2: 0, 3: 1}  # 0=left, 1=right
    to_side_fr = {0: 0, 1: 0, 2: 1, 3: 1}  # 0=front, 1=rear
    
    rel_idx = {0: {1: 60, 2: -90, 3: 150},
               1: {0: -60, 2: -150, 3: 90},
               2: {0: 90, 1: 150, 3: -120},
               3: {0: -150, 1: -90, 2: 120}}
    
    idx_to_degree = {0: 330, 1: 30, 2: 240, 3: 120}  # FL, FR, RL, RR
    visited = set()
    final = []
    combined = fl_events + fr_events + rl_events + rr_events
    
    # Get confidences for direction calculation
    confidences = [combined[i].get("confidence", 0.0) for i in range(4)]
    
    for i in range(len(combined)):
        if i not in visited:
            curr = combined[i]["event"]
            indices = get_indices_from_dict(curr, combined)
            visited.update(indices)
            
            # Don't drop if all 4 are the same - use confidence to find direction
            if len(indices) == 4:
                # All channels have same event - use confidence-weighted direction
                total_conf = sum(confidences) + 1e-8
                if total_conf < 0.04:  # All silence
                    return []
                
                # Weighted average degree
                weighted_degree = sum(idx_to_degree[j] * confidences[j] for j in range(4)) / total_conf
                weighted_degree = weighted_degree % 360.0
                
                # Use highest confidence event
                chosen_idx = int(np.argmax(confidences))
                result = combined[chosen_idx].copy()
                result["degree"] = float(weighted_degree)
                return result
            
            if len(indices) > 1:
                # Find channel with highest confidence
                max_conf = -1
                chosen = None
                for idx in indices:
                    conf = confidences[idx]
                    if conf > max_conf:
                        max_conf = conf
                        chosen = idx
                
                chosen_conf = confidences[chosen]
                chosen_lr_side = to_side_lr[chosen]
                chosen_fr_side = to_side_fr[chosen]
                
                if len(indices) == 2:
                    other = [idx for idx in indices if idx != chosen][0]
                    other_conf = confidences[other]
                    total_conf = chosen_conf + other_conf + 1e-8
                    chosen_norm = chosen_conf / total_conf
                    other_norm = other_conf / total_conf
                    
                    if to_side_lr[other] == chosen_lr_side or to_side_fr[other] == chosen_fr_side:
                        # Same side - interpolate degrees
                        chosen_degree = idx_to_degree[chosen]
                        other_degree = idx_to_degree[other]
                        
                        if chosen_fr_side == 0:  # front
                            if other_degree < chosen_degree:
                                other_degree += 360.0
                            else:
                                chosen_degree += 360.0
                        
                        weighted_degree = ((chosen_degree * chosen_norm + other_degree * other_norm)) % 360.0
                        combined[chosen]["degree"] = float(weighted_degree)
                        final.append([combined[chosen]])
                    
                    elif other_conf > chosen_conf * 0.70:
                        # Opposite corners with similar confidence - use weighted average
                        chosen_degree = idx_to_degree[chosen]
                        other_degree = idx_to_degree[other]
                        weighted_degree = ((chosen_degree * chosen_norm + other_degree * other_norm)) % 360.0
                        combined[chosen]["degree"] = float(weighted_degree)
                        final.append([combined[chosen]])
                    else:
                        # Opposite corner with low confidence other
                        combined[chosen]["degree"] = float(idx_to_degree[chosen])
                        final.append([combined[chosen]])
                
                if len(indices) == 3:
                    front = 0
                    back = 0
                    left = 0
                    right = 0
                    for idx in indices:
                        side_lr = to_side_lr[idx]
                        side_fr = to_side_fr[idx]
                        conf = confidences[idx]
                        if side_fr == 0:
                            front += conf
                        else:
                            back += conf
                        if side_lr == 0:
                            left += conf
                        else:
                            right += conf
                    total_conf = front + back + left + right + 1e-8
                    front = front / total_conf
                    back = back / total_conf
                    left = left / total_conf
                    right = right / total_conf
                    degree = azimuth_from_three_energy(front, back, left, right)
                    combined[chosen]["degree"] = degree
                    final.append([combined[chosen]])
            else:
                combined[i]["degree"] = float(idx_to_degree[i])
                final.append(combined[i])
    
    return final


def combine_quadro_azimuth(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel):


    combined = fl_events + fr_events + rl_events + rr_events

    fl_channel = _to_numpy(fl_channel)
    fr_channel = _to_numpy(fr_channel)
    rl_channel = _to_numpy(rl_channel)
    rr_channel = _to_numpy(rr_channel)

    FL = _energy(fl_channel)
    FR = _energy(fr_channel)
    RL = _energy(rl_channel)
    RR = _energy(rr_channel)

    
    #if no_events: # without events cant distinguish , no if as this is the case for this func call
    if within_margin([FL,FR,RL,RR],margin=0.25):
        print("All energies within margin, no dominant direction, returning no events")
        return []

    #print(f"Energies - FL: {FL}, FR: {FR}, RL: {RL}, RR: {RR}")
    # your symmetric surround formula
    y = (FR + FL) - (RL + RR)
    x = (FR + RR) - (FL + RL)

    theta = np.arctan2(y, x)
    theta_deg = np.degrees(theta)

    if theta_deg < 0:
        theta_deg += 360.0
    
    theta_deg = theta_deg +90.0
    theta_deg = theta_deg % 360.0
    
    _, quadrant_idx = quadro_quadrant_from_azimuth(theta_deg)

    final = combined[quadrant_idx]
    final["degree"] = float(theta_deg)

    return final

# asimuth isnt really combining, more argmax energy direction

def combine_quadro_azimuth_logic(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel):
    to_side_lr = {0: 0, 1: 1, 2: 0, 3: 1}
    to_side_fr = {0: 0, 1: 0, 2: 1, 3: 1}

    idx_to_degree = {0: 330, 1: 30, 2: 240, 3: 120} # sr= 90, sl =270
    visited = set()
    final = []
    combined = fl_events + fr_events + rl_events + rr_events    

    fl_channel = _to_numpy(fl_channel)
    fr_channel = _to_numpy(fr_channel)
    rl_channel = _to_numpy(rl_channel)
    rr_channel = _to_numpy(rr_channel)
    
    combined_vals = [fl_channel, fr_channel, rl_channel, rr_channel]
    for i in range(len(combined)):
        if i not in visited:
            curr = combined[i]["event"]
            indices = get_indices_from_dict(curr,combined)
            visited.update(indices) # since these are all dupe (might be just 1), visited

            if len(indices) == 4:
                print("All channels have the same event, dropping all")
                return [] # all same, drop all

            if len(indices) > 1:
                max_energy = -1
                chosen = None
                for idx in indices:
                    energy = _energy(combined_vals[idx]) 
                    if energy > max_energy:
                        max_energy = energy
                        chosen = idx

                chosen_energy = _energy(combined_vals[chosen])

                chosen_lr_side = to_side_lr[chosen]
                chosen_fr_side = to_side_fr[chosen]
                
                if len(indices) == 2:
                    other = [idx for idx in indices if idx != chosen][0]
                    other_energy = _energy(combined_vals[other])
                    total_energy = chosen_energy + other_energy
                    chosen_energy = chosen_energy / total_energy
                    other_energy = other_energy / total_energy

                    if to_side_lr[other] == chosen_lr_side or to_side_fr[other] == chosen_fr_side: 
                        chosen_degree = idx_to_degree[chosen]
                        other_degree = idx_to_degree[other]

                        if chosen_fr_side == 0: # front
                            if other_degree < chosen_degree:
                                other_degree += 360.0
                            else:
                                chosen_degree += 360.0

                        weighted_degree = ( (chosen_degree * chosen_energy + other_degree * other_energy) / (chosen_energy + other_energy) ) % 360.0
                        #print('summary of combination:')
                        #print(f"chosen_degree: {chosen_degree}, other_degree: {other_degree}")
                        #print(f"chosen_energy: {chosen_energy}, other_energy: {other_energy}")
                        #print(f"weighted_degree: {weighted_degree}")
                        combined[chosen]["degree"] = float(weighted_degree)
                        final.append([combined[chosen]]) 

                    elif other_energy > chosen_energy * 0.70: # only drop if other energy high enough
                        final.append([]) # opposite corners -> no mutual info (), drop both
                    else: # opp corner case with one low energy
                        final.append([combined[chosen]]) 
                
                if len(indices) == 3:
                    front = 0
                    back = 0
                    left = 0
                    right = 0
                    for idx in indices:
                        side_lr = to_side_lr[idx]
                        side_fr = to_side_fr[idx]
                        energy = _energy(combined_vals[idx])
                        if side_fr == 0:
                            front += energy
                        else:
                            back += energy
                        if side_lr == 0:
                            left += energy
                        else:
                            right += energy
                    total_energy = front + back + left + right
                    front = front / total_energy
                    back = back / total_energy
                    left = left / total_energy
                    right = right / total_energy
                    degree = azimuth_from_three_energy(front, back, left, right)
                    combined[chosen]["degree"] = degree
                    final.append([combined[chosen]])
            else:
                final.append(combined[i])

    return final



def combine_quadro_weighted(fl_events, fr_events, rl_events, rr_events, fl_channel, fr_channel, rl_channel, rr_channel):
    to_side_lr = {0: 0, 1: 1, 2: 0, 3: 1}
    to_side_fr = {0: 0, 1: 0, 2: 1, 3: 1}
    temperature = 0.7 # todo, arg, tune
    
    # seperate opposites

    rel_idx = {0: {1:60, 2:-90, 3:150}, # relative degrees 
               1: {0:-60, 2:-150, 3:90},
               2: {0:90, 1:150, 3:-120},
               3: {0:-150, 1:-90, 2:120}}

    idx_to_degree = {0: 330, 1: 30, 2: 240, 3: 120} # sr= 90, sl =270
    visited = set()
    final = []
    combined = fl_events + fr_events + rl_events + rr_events    

    fl_channel = _to_numpy(fl_channel)
    fr_channel = _to_numpy(fr_channel)
    rl_channel = _to_numpy(rl_channel)
    rr_channel = _to_numpy(rr_channel)
    
    combined_vals = [fl_channel, fr_channel, rl_channel, rr_channel]
    for i in range(len(combined)):
        if i not in visited:
            curr = combined[i]["event"]
            indices = get_indices_from_dict(curr,combined)
            visited.update(indices) # since these are all dupe (might be just 1), visited

            if len(indices) == 4:
                print("All channels have the same event, dropping all")
                return [] # all same, drop all
            
            energies = {}
            for idx in indices:
                energies[idx] = _energy(combined_vals[idx])  # softmax
            softmax_with_temp_vals = softmax_with_temp(list(energies.values()),temp=temperature)
            
            for idx, val in zip(energies.keys(), softmax_with_temp_vals):
                energies[idx] = val
            
            main_idx = max(energies, key=energies.get)
            curr_degree = idx_to_degree[main_idx] * energies[main_idx]

            for idx in indices:
                if idx != main_idx:
                    rel_degree = rel_idx[main_idx][idx]
                    real_degree = (idx_to_degree[main_idx] + rel_degree)  
                    curr_degree += real_degree * energies[idx]
            curr_degree = curr_degree % 360.0
            combined[main_idx]["degree"] = float(curr_degree)
            final.append([combined[main_idx]])
            #print("summary of combination:")
            #print(f"indices: {indices}")
            #print(f"energies: {energies}")
            #print(f"final_degree: {curr_degree}")
            
    return final


# -------- 7.1 -------

def combine_7one_azimuth_logic(
    fl_events, fr_events, rl_events, rr_events, sl_events, sr_events,
    fl_channel, fr_channel, rl_channel, rr_channel, sl_channel, sr_channel
):
    # Index mapping for 7.1 directional channels 
    # 0: FL, 1: FR, 2: RL, 3: RR, 4: SL, 5: SR
    idx_to_degree = {
        0: 330,  # FL
        1: 30,   # FR
        2: 210,  # RL
        3: 150,  # RR
        4: 270,  # SL
        5: 90,   # SR
    }
    to_side_lr = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}
    to_side_fr = {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1}

    fl_channel = _to_numpy(fl_channel)
    fr_channel = _to_numpy(fr_channel)
    rl_channel = _to_numpy(rl_channel)
    rr_channel = _to_numpy(rr_channel)
    sl_channel = _to_numpy(sl_channel)
    sr_channel = _to_numpy(sr_channel)

    channels = [fl_channel, fr_channel, rl_channel, rr_channel, sl_channel, sr_channel]
    combined = fl_events + fr_events + rl_events + rr_events + sl_events + sr_events

    visited = set()
    final = []

    for i in range(len(combined)):
        if i not in visited:
            curr = combined[i]["event"]
            indices = get_indices_from_dict(curr, combined)
            visited.update(indices)

            if len(indices) > 4:
                print("5 or 6 /6  channels have the same event, dropping all") # todo, sta toe en weigh
                return []

            if len(indices) > 1:
                max_energy = -1
                chosen = None
                for idx in indices:
                    energy = _energy(channels[idx])
                    if energy > max_energy:
                        max_energy = energy
                        chosen = idx

                chosen_energy = _energy(channels[chosen])
                chosen_lr_side = to_side_lr[chosen]
                chosen_fr_side = to_side_fr[chosen]

                if len(indices) == 2:
                    other = [idx for idx in indices if idx != chosen][0]
                    other_energy = _energy(channels[other])
                    total_energy = chosen_energy + other_energy
                    chosen_energy = chosen_energy / total_energy
                    other_energy = other_energy / total_energy

                    if to_side_lr[other] == chosen_lr_side or to_side_fr[other] == chosen_fr_side:
                        chosen_degree = idx_to_degree[chosen]
                        other_degree = idx_to_degree[other]
                        if chosen_fr_side == 0:
                            if other_degree < chosen_degree:
                                other_degree += 360.0
                            else:
                                chosen_degree += 360.0
                        weighted_degree = (
                            (chosen_degree * chosen_energy + other_degree * other_energy)
                            / (chosen_energy + other_energy)
                        ) % 360.0
                        combined[chosen]["degree"] = float(weighted_degree)
                        final.append([combined[chosen]])
                    elif other_energy > chosen_energy * 0.70:
                        final.append([])
                    else:
                        final.append([combined[chosen]])

                if len(indices) == 3 or len(indices) == 4: # treat 4 exactly same as we collapse to 2 axes eitherway
                    front = back = left = right = 0
                    for idx in indices:
                        side_lr = to_side_lr[idx]
                        side_fr = to_side_fr[idx]
                        energy = _energy(channels[idx])
                        if side_fr == 0:
                            front += energy
                        else:
                            back += energy
                        if side_lr == 0:
                            left += energy
                        else:
                            right += energy
                    total_energy = front + back + left + right
                    front /= total_energy
                    back /= total_energy
                    left /= total_energy
                    right /= total_energy
                    degree = azimuth_from_three_energy(front, back, left, right)
                    combined[chosen]["degree"] = degree
                    final.append([combined[chosen]])
            else:
                final.append(combined[i])

    return final

def combine_7one_azimuth(
    fl_events, fr_events, rl_events, rr_events, sl_events, sr_events,
    fl_channel, fr_channel, rl_channel, rr_channel, sl_channel, sr_channel):
    """
    Returns surround azimuth ranged 0–360 for 7.1 (6 directional channels).
    """
    combined = fl_events + fr_events + rl_events + rr_events + sl_events + sr_events

    fl_channel = _to_numpy(fl_channel)
    fr_channel = _to_numpy(fr_channel)
    rl_channel = _to_numpy(rl_channel)
    rr_channel = _to_numpy(rr_channel)
    sl_channel = _to_numpy(sl_channel)
    sr_channel = _to_numpy(sr_channel)

    FL = _energy(fl_channel)
    FR = _energy(fr_channel)
    RL = _energy(rl_channel)
    RR = _energy(rr_channel)
    SL = _energy(sl_channel)
    SR = _energy(sr_channel)


    if within_margin([FL, FR, RL, RR, SL, SR], margin=0.35):
        print("All energies within margin, no dominant direction, returning no events")
        return []

    #print(f"Energies - FL: {FL}, FR: {FR}, RL: {RL}, RR: {RR}, SL: {SL}, SR: {SR}")
    
    # Y: Front - Back
    y = (FL + FR) - (RL + RR)
    #want to do maybe remove bias caused by SL,SR, s.t. for high vals of these we dont trust the y as much
    
    # X: Right - Left
    x = (FR + SR + RR) - (FL + SL + RL)

    theta = np.arctan2(y, x)
    theta_deg = np.degrees(theta)

    if theta_deg < 0:
        theta_deg += 360.0
    
    theta_deg = theta_deg + 90.0
    theta_deg = theta_deg % 360.0
    
    _, best_idx = seven_one_quadrant_from_azimuth(theta_deg)

    final = combined[best_idx]
    final["degree"] = float(theta_deg)

    return final

# todo "normalise" func to remove clutter
# todo: instead of y,x azimuth based used weighted angle -> less mathmetically correct but probably more adaptive/realistic