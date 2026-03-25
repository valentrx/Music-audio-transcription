import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import glob
from music21 import converter, stream, instrument, metadata, environment, tempo
from PIL import Image
import subprocess


def get_midi_name(midi):
    if isinstance(midi, str): 
        return os.path.splitext(os.path.basename(midi))[0].capitalize()
    elif hasattr(midi, 'filename') and midi.filename: 
        return os.path.splitext(os.path.basename(midi.filename))[0].capitalize()
    else:
        return "Part" 


def add_margins_and_white_bg(image_path, top=20, right=60, bottom=20, left=60):
    img = Image.open(image_path).convert("RGBA")
    new_width = img.width + left + right
    new_height = img.height + top + bottom
    background = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 255))
    background.paste(img, (left, top), mask=img)
    background.convert("RGB").save(image_path, "PNG")


def fix_all_images(output_dir, png_prefix="score"):
    png_prefix = os.path.join(output_dir, png_prefix)
    png_files = sorted(glob.glob(f"{png_prefix}-*.png"))
    if os.path.exists(f"{png_prefix}.png"):
        png_files.insert(0, f"{png_prefix}.png")
    for path in png_files:
        add_margins_and_white_bg(path)


def pngs_to_pdf(png_dir, pdf_path, png_prefix="score"):
    png_prefix = os.path.join(png_dir, png_prefix)
    png_files = sorted(glob.glob(f"{png_prefix}*.png"))

    images = [Image.open(f).convert("RGB") for f in png_files]

    first_image = images[0]
    other_images = images[1:]

    first_image.save(
        pdf_path, save_all=True, append_images=other_images
    )


def midi_treatment(midi_file, output_dir):
    us = environment.UserSettings()
    us['musicxmlPath'] = '/usr/bin/mscore3'
    us['musescoreDirectPNGPath'] = '/usr/bin/mscore3'

    score = converter.parse(midi_file)

    tempi = list(score.recurse().getElementsByClass(tempo.MetronomeMark))
    if len(tempi) > 1:
        for tm in tempi[1:]:
            parent = tm.getContextByClass('Measure')
            if parent:
                parent.remove(tm)
            else:
                tm.activeSite.remove(tm)
    # ================= Convert to PDF directly via musicxml=============
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, "score.png")
    # score.write('musicxml.png', fp=output_path)

    # fix_all_images(output_dir)

    # output_pdf = os.path.join(output_dir, "score.pdf")
    # pngs_to_pdf(output_dir, output_pdf)
    # ======================================================================
    musicxml_path = os.path.join(output_dir, "score.musicxml")
    score.write('musicxml', fp=musicxml_path)

    # Call MuseScore via CLI to convert XML to PDF without GUI
    pdf_path = os.path.join(output_dir, "score.pdf")
    try:
        subprocess.run(['xvfb-run', '--auto-servernum', '--server-args=-screen 0 640x480x24',str(us['musicxmlPath']), musicxml_path, '-o', pdf_path], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MuseScore PDF generation failed:\n{e}")
    

def multi_midi_treatment(midi_dir, output_dir, original_filename=None):
    us = environment.UserSettings()
    us['musicxmlPath'] = '/usr/bin/mscore3'
    us['musescoreDirectPNGPath'] = '/usr/bin/mscore3'

    full_score = stream.Score()
    full_score.metadata = metadata.Metadata()
    
    midi_files = sorted(glob.glob(os.path.join(midi_dir, "*.mid")))
    if not midi_files:
        raise ValueError(f"No MIDI files found in {midi_dir}")
    
    if original_filename != None:
        full_score.metadata.title = original_filename
    else :
        full_score.metadata.title = " + ".join([get_midi_name(f) for f in midi_files])

    for midi_file in midi_files:
        parsed_midi = converter.parse(midi_file)
        part = parsed_midi.parts[0] if isinstance(parsed_midi, stream.Score) else parsed_midi

        for instr in part.recurse().getElementsByClass('Instrument'):
            instr.activeSite.remove(instr)

        name_guess = get_midi_name(midi_file)
        instr = instrument.Instrument()
        instr.partName = name_guess
        instr.partAbbreviation = name_guess[:3].capitalize()
        instr.instrumentName = name_guess 
        part.insert(0, instr)

        full_score.append(part)

    # Remove duplicate tempi
    tempi = list(full_score.recurse().getElementsByClass(tempo.MetronomeMark))
    if len(tempi) > 1:
        for tm in tempi[1:]:
            parent = tm.getContextByClass('Measure')
            if parent:
                parent.remove(tm)
            else:
                tm.activeSite.remove(tm)

    os.makedirs(output_dir, exist_ok=True)

    musicxml_path = os.path.join(output_dir, "score.musicxml")
    full_score.write('musicxml', fp=musicxml_path)

    pdf_path = os.path.join(output_dir, "score.pdf")
    try:
        subprocess.run([
            'xvfb-run', '--auto-servernum', '--server-args=-screen 0 640x480x24',
            str(us['musicxmlPath']), musicxml_path, '-o', pdf_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MuseScore PDF generation failed:\n{e}")
    