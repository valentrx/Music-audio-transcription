import os, re, tempfile, shutil, subprocess, uuid, threading, time, zipfile
from flask import Flask, request, send_file, Response, jsonify
from flask_cors import CORS
from io import BytesIO
from pyngrok import ngrok
from src.run_model import process_audio
from src.progress_tracker import progress_tracker

app = Flask(__name__)
CORS(app)

@app.after_request
def add_ngrok_header(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

@app.route('/convert', methods=['POST'])
def convert():
    file = request.files.get('audio')
    youtube_url = request.form.get('youtube_url')
    format = request.form.get('format')  # 'midi', 'pdf', or 'both'

    if not file and not youtube_url:
        return {'error': 'No file uploaded or YouTube link provided'}, 400

    # Generate unique session ID for progress tracking
    session_id = str(uuid.uuid4())
    output_dir = tempfile.mkdtemp()

    if file:
        # Handle uploaded audio file
        original_filename = os.path.splitext(file.filename)[0] if file.filename else "converted"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_input:
            file.save(tmp_input.name)
            input_path = tmp_input.name
    elif youtube_url:
        # Handle YouTube link download
        original_filename = "youtube_audio"
        input_path = os.path.join(output_dir, "input.wav")
        try:
            subprocess.run([
                "yt-dlp",
                "-x", "--audio-format", "wav","--no-playlist",
                "--output", input_path,
                youtube_url
            ], check=True)
        except subprocess.CalledProcessError as e:
            return {'error': f'Failed to download from YouTube: {e}'}, 500

    def process_in_background():
        try:
            process_audio(input_path, output_dir, format, session_id, original_filename)
        except Exception as e:
            progress_tracker.error_session(session_id, str(e))
        finally:
            pass  # Cleanup is handled later

    # Start background thread
    thread = threading.Thread(target=process_in_background)
    thread.start()

    return jsonify({
        'session_id': session_id,
        'status': 'processing_started'
    })

@app.route('/progress/<session_id>')
def get_progress(session_id):
    """Server-Sent Events endpoint for progress updates"""
    def generate():
        while True:
            progress_data = progress_tracker.get_progress(session_id)
            if progress_data is None:
                yield f"data: {{'error': 'Session not found'}}\n\n"
                break
            
            import json
            yield f"data: {json.dumps(progress_data)}\n\n"
            
            if progress_data['status'] in ['completed', 'error']:
                break
                
            time.sleep(1)  # Update every second
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<session_id>')
def download_result(session_id):
    """Download the converted file"""
    progress_data = progress_tracker.get_progress(session_id)
    
    if not progress_data or progress_data['status'] != 'completed':
        return {'error': 'File not ready or session not found'}, 404
    
    output_dir = progress_data.get('output_dir')
    format_type = progress_data.get('format')
    original_filename = progress_data.get('original_filename', 'converted')
    
    if not output_dir:
        return {'error': 'Output directory not found'}, 404
    
    if format_type == 'both':
        # Create a zip file containing both MIDI and PDF
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add combined.mid file (renamed to original_filename.mid)
            midi_dir = os.path.join(output_dir, 'midi')
            combined_midi_path = os.path.join(midi_dir, 'combined.mid')
            if os.path.exists(combined_midi_path):
                zip_file.write(combined_midi_path, f'{original_filename}.mid')
            
            # Add score.pdf file (renamed to original_filename.pdf)
            score_dir = os.path.join(output_dir, 'score')
            score_pdf_path = os.path.join(score_dir, 'score.pdf')
            if os.path.exists(score_pdf_path):
                zip_file.write(score_pdf_path, f'{original_filename}.pdf')
        
        zip_buffer.seek(0)
        
        # Cleanup
        progress_tracker.cleanup_session(session_id)
        shutil.rmtree(output_dir, ignore_errors=True)
        
        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name=f"{original_filename}.zip",
            mimetype='application/zip'
        )
    
    else:
        # Original logic for single format
        for format_check, ext, subdir in [('midi', 'mid', 'midi'), ('pdf', 'pdf', 'score')]:
            if format_type == format_check:
                search_dir = os.path.join(output_dir, subdir)
                if os.path.exists(search_dir):
                    output_files = [f for f in os.listdir(search_dir) if f.endswith(f".{ext}")]
                    if output_files:
                        output_path = os.path.join(search_dir, output_files[0])
                        
                        with open(output_path, 'rb') as f:
                            file_data = BytesIO(f.read())
                        
                        # Cleanup
                        progress_tracker.cleanup_session(session_id)
                        shutil.rmtree(output_dir, ignore_errors=True)
                        
                        return send_file(
                            file_data,
                            as_attachment=True,
                            download_name=f"{original_filename}.{ext}",
                            mimetype='application/octet-stream'
                        )
    
    return {'error': 'No output file found'}, 500

@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_file(f'static/{path}')

if __name__ == '__main__':
    public_url = ngrok.connect(5000).public_url
    print(f" * ngrok tunnel accessible at : {public_url}")

    # Replace the fetch URL in the JavaScript file
    js_path = 'static/script.js'
    with open(js_path, 'r', encoding='utf-8') as f:
        js_content = f.read()

    pattern = r"fetch\(['\"](.*?/)?convert['\"]"
    replacement = f"fetch('{public_url}/convert'"
    new_js_content = re.sub(pattern, replacement, js_content)

    with open(js_path, 'w', encoding='utf-8') as f:
        f.write(new_js_content)

    app.run(port=5000)
