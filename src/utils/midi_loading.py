import pretty_midi 
import pandas as pd
def extract_notes_(midi_data: pretty_midi.PrettyMIDI) -> dict[str, list]:
    """
    Extract notes from a PrettyMIDI object.

    Args:
        midi_data (pretty_midi.PrettyMIDI): A PrettyMIDI object representing the MIDI file.

    Returns:
        list[pretty_midi.Note]: A list of Note objects extracted from the MIDI file.
    """
    notes_by_instrument = {}
    for instrument in midi_data.instruments:
        if not instrument.is_drum:  # Exclude drum tracks
            name = instrument.name if instrument.name else "Unnamed"
            # print(f"Instrument: {name}, Program: {instrument.program}")
            # Extract notes from the instrument
            notes_by_instrument[name] = {
                "pitch": [],
                "velocity": [],
                "start": [],
                "end": [],
            }
            for note in instrument.notes:
                notes_by_instrument[name]["pitch"].append(note.pitch)
                notes_by_instrument[name]["velocity"].append(note.velocity) 
                notes_by_instrument[name]["start"].append(note.start)
                notes_by_instrument[name]["end"].append(note.end)
    if not notes_by_instrument[name]["pitch"]:
        raise ValueError("No notes found in the MIDI file.")

    return notes_by_instrument

def load_midi(midi_path:str) -> pretty_midi.PrettyMIDI:
    """
    Load a MIDI file and return a PrettyMIDI object.

    Args:
        midi_path (str): Path to the MIDI file.

    Returns:
        pretty_midi.PrettyMIDI: A PrettyMIDI object representing the MIDI file.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        return midi_data
    except Exception as e:
        raise ValueError(f"Error loading MIDI file: {e}")
    
def get_midi_instrument_names(midi_data: pretty_midi.PrettyMIDI) -> list[str]:
    """
    Get the names of instruments in a MIDI file.

    Args:
        midi_data (pretty_midi.PrettyMIDI): A PrettyMIDI object representing the MIDI file.

    Returns:
        list[str]: A list of instrument names.
    """
    return [instrument.name for instrument in midi_data.instruments if instrument.is_drum is False]

def extract_all_notes(midi_data: pretty_midi.PrettyMIDI) -> list[dict]:
    """
    Extract notes from a PrettyMIDI object.

    Args:
        midi_data (pretty_midi.PrettyMIDI): A PrettyMIDI object representing the MIDI file.

    Returns:
        list[dict]: A list of dictionaries containing note information (pitch, velocity, start, end).
    """
    all_notes = []
    print(f"Extracting Notes ...")
    for instrument in midi_data.instruments:
        if not instrument.is_drum:  # Exclude drum tracks
            name = instrument.name if instrument.name else "Unnamed"

            #print(f"Instrument: {name}, Program: {instrument.program}")
            for note in instrument.notes:
                all_notes.append({
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "start": note.start,
                    "end": note.end
                })
    if not all_notes:
        raise ValueError("No notes found in the MIDI file.")
    all_notes.sort(key=lambda x: x['start'])  # Sort notes by start time
    #
    # print(all_notes)
    return all_notes

def export_notes_to_csv(notes: list[dict], csv_path: str):
    """
    Export notes to a CSV file.

    Args:
        notes (list[dict]): A list of dictionaries containing note information (pitch, velocity, start, end).
        csv_path (str): Path to the output CSV file.
    """

    df = pd.DataFrame(notes)
    df.to_csv(csv_path, index=False, sep=";", decimal=",")
    print(f"Notes exported to {csv_path}")


if __name__ == "__main__":
    # Example usage
    # midi_path = 'data/raw/busoni_sonata/Busoni_sonata_no2_op_8-BV_61_Scherzo.mid'
    # midi_data = load_midi(midi_path)
    # # Extract notes
    # notes = extract_all_notes(midi_data)
    
    # # Export notes to CSV
    # export_notes_to_csv(notes, 'output/notes.csv')
    
    # # Print instrument names
    # instrument_names = get_midi_instrument_names(midi_data)
    # print("Instruments in MIDI file:", instrument_names)

    
    midi_gt = load_midi('data/raw/busoni_sonata/Busoni_sonata_no2_op_8-BV_61_Scherzo.mid')
    midi_prd = load_midi('output/model_midi/experiment13/tmp_pred_27.mid')

    notes_gt = extract_all_notes(midi_gt)
    notes_prd = extract_all_notes(midi_prd)

    export_notes_to_csv(notes_gt, 'output/model_midi/experiment13/gt.csv')
    export_notes_to_csv(notes_prd, 'output/model_midi/experiment13/trail_27_pred.csv')