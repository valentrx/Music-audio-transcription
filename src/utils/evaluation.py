from src.utils import midi_loading as ml
import numpy as np
import librosa 

def f1_score(precision: float, recall: float) -> float:
    """
    Calculate the F1 score given precision and recall.

    Args:
        precision (float): The precision value.
        recall (float): The recall value.

    Returns:
        float: The F1 score.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def f1_score_with_overlap(predicted_notes: dict, ground_truth_notes: dict, tolerance:float=3, min_overlap:float=0.25) -> dict:
    """    
        Calculate the F1 score with overlap tolerance for predicted and ground truth notes.
        This function compares predicted notes with ground truth notes, allowing for a specified tolerance in start and end times,
        and a minimum overlap duration to consider a note as a true positive.

        Default tolerance is set to 0.05 seconds and minimum overlap to 0.5 seconds.

        Args:

        predicted_notes (dict): Dictionary of predicted notes with keys 'pitch', 'start', 'end', and 'instrument'.
        ground_truth_notes (dict): Dictionary of ground truth notes with keys 'pitch', 'start', 'end', and 'instrument'.
        tolerance (float): Tolerance for overlap in seconds.        
        min_overlap (float): Minimum overlap required to consider a note as true positive. given as a percentage of the shorter note's duration.

    Returns:
        dict: A dictionary containing the F1 score, precision, and recall.
    """

    try:
        # Initialize counters for true positives, false positives, and false negatives
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        # Iterate through predicted instruments
        # and match with ground truth instruments

        # Iterate through predicted notes and match with ground truth notes
        for pred_note in predicted_notes:
            # export to csv file

            matched = False
            for gt_note in ground_truth_notes:
                if (
                    pred_note['pitch'] == gt_note['pitch'] #and
                    #abs(pred_note['start'] - gt_note['start']) <= tolerance #and
                    #abs(pred_note['end'] - gt_note['end']) <= tolerance
                ):
                    # print("================================================")
                    # print(f"Matching Predicted Note: {pred_note}")
                    # print(f"Matching Ground Truth Note: {gt_note}")
                    # Check if the notes overlap
                    # If the start of the predicted note is before the end of the ground truth note
                    # and the end of the predicted note is after the start of the ground truth note
                    # This means the notes overlap
                    overlap = min(pred_note['end'], gt_note['end']) - max(pred_note['start'], gt_note['start'])
                    # print(f"Overlap: {overlap}")
                    # If the overlap is greater than or equal to the minimum required overlap   
                    # Calculate the minimum required overlap as a percentage of the shorter note's duration
                    min_note_duration = min(pred_note['end'] - pred_note['start'], gt_note['end'] - gt_note['start'])
                    required_overlap = min_note_duration * min_overlap
                    if overlap >= required_overlap:
                        true_positives += 1
                        matched = True
                        break
                    #true_positives += 1
                    #print(f"+1")
                    #matched = True
                    #break
            if not matched:
                false_positives += 1

        false_negatives = len(ground_truth_notes) - true_positives
        print(f"True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}")
        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = f1_score(precision, recall)
        if f1 is None:
            raise ValueError("F1 score could not be calculated.")
        
        return dict(
            f1_score=f1,
            precision=precision,
            recall=recall,
            false_negatives=false_negatives,
            false_positives=false_positives,
            true_positives=true_positives
        )
    
    except Exception as e:
        raise ValueError(f"Error calculating F1 score with overlap and instruments: {e}")

def align_notes(predicted_notes:dict, ground_truth_notes:dict, max_offset=2.0, step=0.01)-> tuple[float, float, dict]:
    """
    Align the MIDI File to the Ground Truth File to achieve maximum F1 Score.

    Args:
        predicted_notes (dict): Dictionary of predicted notes with keys 'pitch', 'start', 'end', and 'instrument'.
        ground_truth_notes (dict): Dictionary of ground truth notes with keys 'pitch', 'start', 'end', and 'instrument'.
        max_offset (float): Maximum allowed note offset
    """
        # Angenommen, beide Listen sind nach Startzeit sortiert:
    offset = notes_gt[0]['start'] - notes_prd[0]['start']
    notes_prd_aligned = [{**note, "start": note["start"] + offset, "end": note["end"] + offset} for note in notes_prd]
    return offset, notes_prd_aligned
def pitch_deviation(pred_notes: list[dict], gt_notes: list[dict], match_window=0.1):
    """
    Computes the average absolute pitch difference between matched predicted and ground truth notes.
    """
    if not pred_notes or not gt_notes:
        return np.nan
    pitch_diffs = []
    for p_note in pred_notes:
        time_diffs = [abs(gt_note['start'] - p_note['start']) for gt_note in gt_notes]
        if not time_diffs:
            continue
        idx = np.argmin(time_diffs)
        if time_diffs[idx] <= match_window:
            pitch_diffs.append(abs(p_note['pitch'] - gt_notes[idx]['pitch']))
    return np.mean(pitch_diffs) if pitch_diffs else np.nan


def onset_deviation(pred_notes: list[dict], gt_notes: list[dict]):
    """
    Computes the average onset deviation between predicted and closest ground truth notes.
    """
    if not pred_notes or not gt_notes:
        return np.nan
    onset_diffs = []
    for p_note in pred_notes:
        time_diffs = [abs(gt_note['start'] - p_note['start']) for gt_note in gt_notes]
        if not time_diffs:
            continue
        idx = np.argmin(time_diffs)
        onset_diffs.append(time_diffs[idx])
    return np.mean(onset_diffs) if onset_diffs else np.nan


def duration_deviation(pred_notes: list[dict], gt_notes: list[dict]):
    """
    Computes the average duration difference between predicted and closest ground truth notes.
    """
    if not pred_notes or not gt_notes:
        return np.nan
    dur_diffs = []
    for p_note in pred_notes:
        time_diffs = [abs(gt_note['start'] - p_note['start']) for gt_note in gt_notes]
        if not time_diffs:
            continue
        idx = np.argmin(time_diffs)
        pred_dur = p_note['end'] - p_note['start']
        gt_dur = gt_notes[idx]['end'] - gt_notes[idx]['start']
        dur_diffs.append(abs(pred_dur - gt_dur))
    return np.mean(dur_diffs) if dur_diffs else np.nan


def density_deviation(pred_notes: list[dict], gt_notes: list[dict], binsize=2.0):
    """
    Computes the average difference in note density over fixed time bins.
    """
    if not pred_notes or not gt_notes:
        return np.nan
    pred_starts = [note['start'] for note in pred_notes]
    gt_starts = [note['start'] for note in gt_notes]
    total_duration = max(max(pred_starts, default=0), max(gt_starts, default=0))
    bins = np.arange(0, total_duration + binsize, binsize)
    gt_hist, _ = np.histogram(gt_starts, bins=bins)
    pred_hist, _ = np.histogram(pred_starts, bins=bins)
    return np.mean(np.abs(gt_hist - pred_hist)) if len(gt_hist) == len(pred_hist) else np.nan



def evaluate_midi(midi_path_pred: str, midi_path_ground_truth: str, tolerance_note: float=0.1, overlap_note: float=0.1) -> dict:
    """
    Evaluate a MIDI file against ground truth data, considering instruments.

    Args:
        midi_path (str): Path to the predicted MIDI file.
        midi_ground_truth_path (str): Path to the ground truth MIDI file.
    Returns:
        dict: A dictionary containing evaluation metrics per instrument.
    """
    try:
        predicted_data = ml.load_midi(midi_path_pred)
        ground_truth_data = ml.load_midi(midi_path_ground_truth)

    except Exception as e:
        raise ValueError(f"Error loading MIDI files: {e}")
    
    try:
        predicted_notes = ml.extract_all_notes(predicted_data)
        ground_truth_notes = ml.extract_all_notes(ground_truth_data)
        #PAUSE = input("Press Enter to continue...")
        # Export notes to CSV files for further analysis
        print('===============================================')
        print('\nExporting CSVs....')
        ml.export_notes_to_csv(predicted_notes, "output/predicted_notes.csv")
        ml.export_notes_to_csv(ground_truth_notes, "output/ground_truth_notes.csv")
        #PAUSE = input("Press Enter to continue...")
        #print(predicted_notes)
        #print(ground_truth_notes)
        if not predicted_notes or not ground_truth_notes:
            raise ValueError("No notes found in one of the MIDI files.")
        
    except Exception as e:
        raise ValueError(f"Error extracting notes from MIDI files: {e}")      

    try:
        f1 = f1_score_with_overlap(predicted_notes, ground_truth_notes, tolerance=tolerance_note, min_overlap=overlap_note)
        metrics = {
        "pitch_deviation": pitch_deviation(predicted_notes, ground_truth_notes),
        "onset_deviation": onset_deviation(predicted_notes, ground_truth_notes),
        "duration_deviation": duration_deviation(predicted_notes, ground_truth_notes),
        "density_deviation": density_deviation(predicted_notes, ground_truth_notes),
}
        return {
            "f1_score": f1['f1_score'],
            "predicted_midi": midi_path_pred,
            "ground_truth": midi_path_ground_truth,
            "precision": f1['precision'],
            "recall": f1['recall'],
            "false_negatives": f1['false_negatives'],
            "false_positives": f1['false_positives'],   
            "true_positives": f1['true_positives'],
            "pitch_deviation": pitch_deviation(predicted_notes, ground_truth_notes),
            "onset_deviation": onset_deviation(predicted_notes, ground_truth_notes),
            "duration_deviation": duration_deviation(predicted_notes, ground_truth_notes),
            "density_deviation": density_deviation(predicted_notes, ground_truth_notes),

        }
    except Exception as e:
        raise ValueError(f"Error calculating F1-Score: {e}")

if __name__ == "__main__":
    # Example usage
    base_test = False
    if base_test:
        midi_path_pred = "output/model_midi/tmp_pred.mid"
        midi_path_ground_truth = "data/raw/busoni_sonata/Busoni_sonata_no2_op_8-BV_61_Scherzo.mid"
        
        try:
            evaluation_result = evaluate_midi("output/model_midi/tmp_pred.mid", "data/raw/busoni_sonata/Busoni_sonata_no2_op_8-BV_61_Scherzo.mid")
            print("Evaluation Result:", evaluation_result)
        except ValueError as e:
            print(f"Error during evaluation: {e}")
    else:
        midi_gt = ml.load_midi('data/raw/busoni_sonata/Busoni_sonata_no2_op_8-BV_61_Scherzo.mid')
        midi_prd = ml.load_midi('output/model_midi/experiment13/tmp_pred_27.mid')

        notes_gt = ml.extract_all_notes(midi_gt)
        notes_prd = ml.extract_all_notes(midi_prd)
        print(f"F1 without align:{f1_score_with_overlap(notes_prd, notes_gt)}")
        off_set, notes_prd_aligned = align_notes(notes_prd, notes_gt)
        print(f"Offset:{off_set}")
        x = f1_score_with_overlap(notes_prd_aligned, notes_gt)
        print(x)
        ml.export_notes_to_csv(notes_prd_aligned, 'output/model_midi/experiment13/trail_27_pred_aligned.csv')

