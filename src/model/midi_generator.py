import os
import pretty_midi
from basic_pitch.inference import predict, predict_and_save, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
from src.model.model_parameter import INSTRUMENT_PARAMS

# Prototype
# predict_and_save(
#     <input-audio-path-list>,
#     <output-directory>,
#     <save-midi>,
#     <sonify-midi>,
#     <save-model-outputs>,
#     <save-notes>,
# )
#model_output, midi_data, note_events = predict("input/audio/file.wav")
def get_instrument_from_filename(filename: str) -> str:
    """
    Extrahiert das Instrument anhand des Dateinamens.
    """
    for key in INSTRUMENT_PARAMS.keys():
        if key in filename.lower():
            return key
    return "other"  # Fallback, falls nichts gefunden

def transcribe_with_optimal_params(wav_path: str, output_dir: str):
    """
    Führt die Transkription mit den zum Instrument passenden Parametern aus.
    """
    instrument = get_instrument_from_filename(wav_path)
    params = INSTRUMENT_PARAMS[instrument]

    output_path = os.path.join(output_dir, f"{instrument}.mid")
    print(f"Found Instrument: {instrument}")
    return predict_midi(
        audio_path=wav_path,
        onset_threshold=params["onset_threshold"],
        frame_threshold=params["frame_threshold"],
        minimum_note_length=params["minimum_note_length"]
    ), instrument

def predict_midi(
    audio_path: str,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.5,
    minimum_note_length: int = 50
) -> str:
    """
    Predicts a MIDI file from an audio input using basic-pitch with custom hyperparameters.

    Args:
        audio_path (str): Path to the audio file.
        output_path (str): Path to save the predicted MIDI file.
        onset_threshold (float): Threshold for detecting note onsets (0.0 - 1.0).
        frame_threshold (float): Threshold for detecting note frames (0.0 - 1.0).
        minimum_note_length (int): Minimum note duration in milliseconds.

    Returns:
        str: Path to the saved MIDI file.
    """
    print(f"🎶 Predicting MIDI for: {audio_path}")
    print(f"Using hyperparameters: onset_threshold={onset_threshold}, frame_threshold={frame_threshold}, min_note_length={minimum_note_length}")

    _, midi_data, _ = predict(
        audio_path=audio_path,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length
    )
    return midi_data

def combine_midis(midi_list: list[pretty_midi.PrettyMIDI], names: list[str]) -> pretty_midi.PrettyMIDI:
    combined = pretty_midi.PrettyMIDI()
    for i, (midi, name) in enumerate(zip(midi_list, names)):
        src_inst = midi.instruments[0]
        new_inst = pretty_midi.Instrument(program=i, name=name)
        new_inst.notes = src_inst.notes
        new_inst.control_changes = src_inst.control_changes
        new_inst.pitch_bends = src_inst.pitch_bends
        combined.instruments.append(new_inst)
    return combined

# run_model.py
