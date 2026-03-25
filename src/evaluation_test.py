# Adds the project root to the path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.utils.evaluation as eval
import mir_eval
import src.utils.midi_loading as ml
from basic_pitch.inference import ICASSP_2022_MODEL_PATH, predict
from src.model.midi_generator import predict_midi
import numpy as np

def main():
    GT_MIDI_PATH = 'data/raw/busoni_sonata/Scherzo_original_gt.mid'
    audio_path =  'data/raw/busoni_sonata/gymnopedie.mp3'
    
    output_dir = 'output/test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_midi = predict_midi(audio_path=audio_path, onset_threshold=0.7, frame_threshold=0.5, minimum_note_length=80)
    PRED_MIDI_PATH = f"{output_dir}/pred.mid"
    
    prediction_midi.write(PRED_MIDI_PATH)
    gt_midi = ml.load_midi(GT_MIDI_PATH)
    gt_midi.write(f"{output_dir}/gt.mid")

    # Evaluate the generated MIDI file
    evaluation_result = eval.evaluate_midi(
        midi_path_pred=PRED_MIDI_PATH,
        midi_path_ground_truth=GT_MIDI_PATH,
        tolerance_note=1,
        overlap_note=0.1    
    )
    
    print(
    f"\033[30mF1-Score:\033[0m \033[1m{evaluation_result['f1_score']}\033[0m, "
    f"\033[31mPitch Deviation:\033[0m \033[1m{evaluation_result['pitch_deviation']}\033[0m, "
    f"\033[34mOnset Deviation:\033[0m \033[1m{evaluation_result['onset_deviation']}\033[0m, "
    f"\033[32mDuration Deviation:\033[0m \033[1m{evaluation_result['duration_deviation']}\033[0m, "
    f"\033[35mDensity Deviation:\033[0m \033[1m{evaluation_result['density_deviation']}\033[0m\n"
    )
    ref_intervals = np.array([[n.start, n.end] for n in gt_midi.instruments[0].notes])
    ref_pitches = np.array([n.pitch for n in gt_midi.instruments[0].notes])

    est_intervals = np.array([[n.start, n.end] for n in prediction_midi.instruments[0].notes])
    est_pitches = np.array([n.pitch for n in prediction_midi.instruments[0].notes])

    mir_eval_scores = mir_eval.transcription.evaluate(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches
    )
    mir_eval_scores = mir_eval.transcription.evaluate(
    ref_intervals, ref_pitches,
    est_intervals, est_pitches
    )

    print("\n\033[36m[mir_eval ALL METRICS]\033[0m")
    for metric_name, value in mir_eval_scores.items():
        print(f"\033[33m{metric_name}:\033[0m \033[1m{value:.4f}\033[0m")

    # Append to your results
    # evaluation_result.update({
    #     'mir_onset_f1': mir_eval_scores['Onset_F1'],
    #     'mir_onset_offset_f1': mir_eval_scores['Onset_Offset_F1'],
    #     'mir_precision': mir_eval_scores['Precision'],
    #     'mir_recall': mir_eval_scores['Recall']
    # })
    # print(
    # f"\033[36m[mir_eval]\033[0m "
    # f"\033[30mOnset F1:\033[0m \033[1m{evaluation_result['mir_onset_f1']:.3f}\033[0m, "
    # f"\033[34mOnset+Offset F1:\033[0m \033[1m{evaluation_result['mir_onset_offset_f1']:.3f}\033[0m, "
    # f"\033[32mPrecision:\033[0m \033[1m{evaluation_result['mir_precision']:.3f}\033[0m, "
    # f"\033[31mRecall:\033[0m \033[1m{evaluation_result['mir_recall']:.3f}\033[0m\n"
    # )

if __name__ == "__main__":
    main()

