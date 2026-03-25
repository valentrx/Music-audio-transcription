import pretty_midi
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_midi_comparison(
    midi_path_gt: str,
    midi_path_pred: str,
    fs: int = 100,
    save_path: str = None
):
    """
    Visualizes piano roll of ground truth and predicted MIDI files side by side.

    Args:
        midi_path_gt (str): Path to ground truth MIDI file.
        midi_path_pred (str): Path to predicted MIDI file.
        fs (int): Frames per second (time resolution).
        save_path (str): If set, saves the figure to this path.
    """
    if not Path(midi_path_gt).exists():
        raise FileNotFoundError(f"Ground truth MIDI file not found: {midi_path_gt}")
    if not Path(midi_path_pred).exists():
        raise FileNotFoundError(f"Predicted MIDI file not found: {midi_path_pred}")

    gt = pretty_midi.PrettyMIDI(midi_path_gt)
    pred = pretty_midi.PrettyMIDI(midi_path_pred)

    roll_gt = gt.get_piano_roll(fs=fs)
    roll_pred = pred.get_piano_roll(fs=fs)

    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axs[0].imshow(roll_gt, aspect='auto', origin='lower', cmap='Greens')
    axs[0].set_title("✅ Ground Truth")
    axs[0].set_ylabel("MIDI Pitch")

    axs[1].imshow(roll_pred, aspect='auto', origin='lower', cmap='Reds')
    axs[1].set_title("🎯 Predicted MIDI")
    axs[1].set_xlabel("Time (frames)")
    axs[1].set_ylabel("MIDI Pitch")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"🎉 Piano roll saved to {save_path}")
    else:
        plt.show()

# Beispiel-Aufruf
if __name__ == "__main__":
    gt_path = "data/raw/busoni_sonata/Busoni_sonata_no2_op_8-BV_61_Scherzo.mid"
    pred_path = "output/model_midi/tmp_pred.mid"
    visualize_midi_comparison(gt_path, pred_path)
