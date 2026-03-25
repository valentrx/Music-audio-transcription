import functools
import os
import sys

sys.path.append(os.path.abspath('mt3'))  # relative to project root

import numpy as np
import tensorflow.compat.v2 as tf

import gin
import jax
import librosa
import note_seq
import t5
import seqio
import t5x

from mt3 import metrics_utils,models, network, note_sequences, preprocessors, spectrograms, vocabularies

SAMPLE_RATE = 16000

class InferenceModel:
    def __init__(self, checkpoint_path, model_type='mt3'):
        if model_type == 'ismir2021':
            num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
        elif model_type == 'mt3':
            num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        gin_files = ['mt3/gin/model.gin', f'mt3/gin/{model_type}.gin']

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {'inputs': self.inputs_length, 'targets': self.outputs_length}
        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins))
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            'targets': seqio.Feature(vocabulary=self.vocabulary),
        }

        self._parse_gin(gin_files)
        self.model = self._load_model()
        self.restore_from_checkpoint(checkpoint_path)

    def _parse_gin(self, gin_files):
        bindings = [
            'from __gin__ import dynamic_registration',
            'from mt3 import vocabularies',
            'VOCAB_CONFIG=@vocabularies.VocabularyConfig()',
            'vocabularies.VocabularyConfig.num_velocity_bins=1'
        ]
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(gin_files, bindings, finalize_config=False)

    def _load_model(self):
        model_config = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_config)
        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features['inputs'].vocabulary,
            output_vocabulary=self.output_features['targets'].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config)
        )

    def restore_from_checkpoint(self, checkpoint_path):
        initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def,
            init_fn=self.model.get_initial_variables,
            input_shapes=self.input_shapes,
            partitioner=self.partitioner)

        restore_cfg = t5x.utils.RestoreCheckpointConfig(path=checkpoint_path, mode='specific', dtype='float32')
        self._predict_fn = self._get_predict_fn(initializer.train_state_axes)
        self._train_state = initializer.from_checkpoint_or_scratch([restore_cfg], init_rng=jax.random.PRNGKey(0))

    @property
    def input_shapes(self):
        return {
            'encoder_input_tokens': (self.batch_size, self.inputs_length),
            'decoder_input_tokens': (self.batch_size, self.outputs_length)
        }

    @functools.lru_cache()
    def _get_predict_fn(self, train_state_axes):
        def _predict(params, batch, decode_rng):
            return self.model.predict_batch_with_aux(params, batch, decoder_params={'decode_rng': None})

        return self.partitioner.partition(
            _predict,
            in_axis_resources=(train_state_axes.params, t5x.partitioning.PartitionSpec('data',), None),
            out_axis_resources=t5x.partitioning.PartitionSpec('data',)
        )

    def __call__(self, audio):
        ds = self.audio_to_dataset(audio)
        ds = self.preprocess(ds)

        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(ds, task_feature_lengths=self.sequence_length)
        model_ds = model_ds.batch(self.batch_size)

        inferences = (tokens for batch in model_ds.as_numpy_iterator() for tokens in self.predict_tokens(batch))

        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
            predictions.append(self.postprocess(tokens, example))

        return metrics_utils.event_predictions_to_ns(predictions, codec=self.codec, encoding_spec=self.encoding_spec)['est_ns']

    def predict_tokens(self, batch, seed=0):
        prediction, _ = self._predict_fn(self._train_state.params, batch, jax.random.PRNGKey(seed))
        return self.vocabulary.decode_tf(prediction).numpy()

    def audio_to_dataset(self, audio):
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors({'inputs': frames, 'input_times': frame_times})

    def _audio_to_frames(self, audio):
        frame_size = self.spectrogram_config.hop_width
        pad_len = frame_size - len(audio) % frame_size
        audio = np.pad(audio, [0, pad_len], mode='constant')
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    def preprocess(self, ds):
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key='inputs',
                additional_feature_keys=['input_times']),
            preprocessors.add_dummy_targets,
            functools.partial(preprocessors.compute_spectrograms, spectrogram_config=self.spectrogram_config)
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    def postprocess(self, tokens, example):
        tokens = self._trim_eos(tokens)
        start_time = example['input_times'][0]
        start_time -= start_time % (1 / self.codec.steps_per_second)
        return {'est_tokens': tokens, 'start_time': start_time, 'raw_inputs': []}

    @staticmethod
    def _trim_eos(tokens):
        tokens = np.array(tokens, np.int32)
        if vocabularies.DECODED_EOS_ID in tokens:
            tokens = tokens[:np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
        return tokens


if __name__ == '__main__':
    audio_path = 'example.wav'  # Use your file path here
    checkpoint_path = 'checkpoints/mt3/'  # Adjust if needed

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    model = InferenceModel(checkpoint_path)
    note_sequence = model(audio)

    out_midi = 'transcribed.mid'
    note_seq.sequence_proto_to_midi_file(note_sequence, out_midi)
    print(f"Saved transcription to: {out_midi}")
