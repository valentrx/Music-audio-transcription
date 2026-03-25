# import utils.split as splt
# import utils.evaluation as ev
# import utils.model_optimizer as opt
# from model.model_parameter import INSTRUMENT_PARAMS
# import model.midi_generator as midi_gen

import os, subprocess

from pathlib import Path
from src.model import midi_generator as midi_gen
from src.data import postprocess as post_proc

# RAW_AUDIO_PATH = 'data/raw/busoni_sonata/Busoni_sonata_no2_op_8-BV_61_Scherzo.mp3'
# PRED_MIDI_PATH = 'output/model_midi/Busoni_sonata_no2_op_8-BV_61_Scherzo_basic_pitch.mid'
# GT_MIDI_PATH = 'data/raw/busoni_sonata/Busoni_sonata_no2_op_8-BV_61_Scherzo.mid'
# TMP_PRED_MIDI_PATH = 'output/model_midi/tmp_pred.mid'


def process_audio(audio_file, output_path, format, session_id=None, original_filename=None):
    from src.progress_tracker import progress_tracker
    
    if session_id:
        progress_tracker.create_session(session_id, 100)
        progress_tracker.update_progress(session_id, 5, "Initializing directories...")
    
    stem_dir = os.path.join(output_path, "stems")
    midi_dir = os.path.join(output_path, "midi")
    score_dir = os.path.join(output_path, "score")

    os.makedirs(stem_dir, exist_ok=True)
    os.makedirs(midi_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)
    
    # Store output_dir in progress data for later cleanup
    if session_id:
        progress_tracker._progress_data[session_id]['output_dir'] = output_path
        progress_tracker._progress_data[session_id]['format'] = format
        if original_filename:
            progress_tracker._progress_data[session_id]['original_filename'] = original_filename

    '''
    Split the main audio into seperate instrument stems
    '''
    if session_id:
        progress_tracker.update_progress(session_id, 15, "Splitting audio into instrument stems...")
        
    #stems = splt.split_audio(audio_file)
    subprocess.run([
        "/content/music_venv310/bin/python3.10", 
        "src/utils/split.py", 
        audio_file, 
        "--remove_drums",
        "--output", stem_dir,
    ], check=True)

    if session_id:
        progress_tracker.update_progress(session_id, 40, "Audio splitting completed, starting MIDI generation...")

    # import tensorflow as tf
    # tf.keras.backend.clear_session()
    midis = []
    instrument_names = []
    
    # Count stems for progress calculation
    stem_files = list(Path(stem_dir).iterdir())
    total_stems = len(stem_files)
    
    for i, wav_path in enumerate(stem_files):
        name = Path(wav_path).stem
        print(f"Predicting MidiFile for:{name}")
        
        if session_id:
            progress_value = 40 + (i * 30 // total_stems)  # 40-70% for MIDI generation
            progress_tracker.update_progress(session_id, progress_value, f"Generating MIDI for: {name}")
        
        '''
        Iterate over all stems and generate .mid + score
        '''
        #wav_path = stem_dir / f"{name}.wav"
        #sf.write(wav_path, wav, 44100)
        # Generate MIDI file for this stem
        midi, instrument = midi_gen.transcribe_with_optimal_params(
            wav_path=str(wav_path),
            output_dir=str(midi_dir)
        )
        midi.write(os.path.join(midi_dir, f"{name}.mid"))
        
        midis.append(midi)
        instrument_names.append(instrument)
        
    if session_id:
        progress_tracker.update_progress(session_id, 75, "Combining MIDI files...")

    if session_id:
        progress_tracker.update_progress(session_id, 85, "Generating score from MIDI...")

    # Generate score from MIDI and save in `score_dir`
    # Always generate PDF for 'both' format, or when specifically requested
    if format in ['pdf', 'both']:
        post_proc.multi_midi_treatment(midi_dir, score_dir, original_filename)

    # Combine all generated MIDI files into one
    midi_gen.combine_midis(midis, instrument_names).write(os.path.join(midi_dir, "combined.mid"))
    
    if session_id:
        progress_tracker.complete_session(session_id)


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to the input audio file')
    parser.add_argument('--output', type=str, required=False, default='output/')
    parser.add_argument('--format', type=str, choices=['midi', 'pdf', 'both'], required=False, default='midi')
    args = parser.parse_args()

    process_audio(args.input, args.output, args.format)
