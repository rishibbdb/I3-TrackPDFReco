import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import itertools

from lib.experimental_methods import get_first_regular_pulse

try:
    import tensorflow as tf
except ImportError:
    print("did not find tensorflow(cpu). can not use batched data loader.")

class I3SimHandler:
    def __init__(self, events_meta_file: str = None,
                 events_pulses_file: str = None,
                 geo_file: str = None,
                 df_meta: pd.DataFrame = None,
                 df_pulses: pd.DataFrame = None) -> None:

        if ((events_meta_file is not None) and
                events_pulses_file is not None):
            self.events_meta = pd.read_feather(events_meta_file)
            self.events_data = pd.read_feather(events_pulses_file)

        else:
            self.events_meta = df_meta
            self.events_data = df_pulses

        self.geo = pd.read_csv(geo_file)

    def get_event_data(self, event_index: int) -> pd.DataFrame:
        ev_idx = event_index
        event_meta = self.events_meta.iloc[ev_idx]
        event_data = (self.events_data.iloc[int(event_meta.idx_start): int(event_meta.idx_end + 1)]).copy(deep=True)
        return event_meta, event_data

    def get_per_dom_summary_from_sim_data(self,
                                          meta: pd.DataFrame,
                                          pulses: pd.DataFrame,
                                          charge_key='charge',
                                          correct_charge=False) -> pd.DataFrame:

        df_qtot = pulses[['sensor_id', charge_key]].groupby(by=['sensor_id'], as_index=False).sum()
        df_tmin = pulses[['sensor_id', 'time']].groupby(by=['sensor_id'], as_index=False).min()
        df = df_qtot.merge(self.geo.iloc[df_qtot['sensor_id']], on='sensor_id', how='outer')
        df['time'] = df_tmin['time'].values

        if correct_charge == True:
            df_corr = pulses[['sensor_id', 'charge_correction']].groupby(by=['sensor_id'], as_index=False).mean()
            df['charge'] = df['charge'].values * df_corr['charge_correction'].values

        if charge_key != 'charge':
            df.rename({charge_key: 'charge'}, inplace=True, axis='columns')
        return df

    def get_per_dom_summary_from_index(self, event_index: int, charge_key='charge', correct_charge=False) -> pd.DataFrame:
        #df_qtot = pulses[['sensor_id', charge_key]].groupby(by=['sensor_id'], as_index=False).sum()
        #df_tmin = pulses[['sensor_id', 'time']].groupby(by=['sensor_id'], as_index=False).min()
        #df = df_qtot.merge(self.geo.iloc[df_qtot['sensor_id']], on='sensor_id', how='outer')
        #df['time'] = df_tmin['time'].values

        #if charge_key != 'charge':
        #    df.rename({charge_key: 'charge'}, inplace=True, axis='columns')
        #return df

        # avoid duplicating code (see above)
        meta, pulses = self.get_event_data(event_index)
        return self.get_per_dom_summary_from_sim_data(meta, pulses, charge_key=charge_key, correct_charge=correct_charge)

    def replace_early_pulse(self, summary_data, pulses):
        corrected_time = np.zeros(len(summary_data))
        for i, row in summary_data.iterrows():
            s_id = row['sensor_id']
            q_tot = row['charge']
            t1 = row['time']

            idx = pulses['sensor_id'] == s_id
            pulses_this_dom = pulses[idx]
            corrected_time[i] = get_first_regular_pulse(pulses_this_dom, t1, q_tot)



        summary_data['time'] = corrected_time

# TODO: add bucket_by_sequence length to optimize padded batches.
# See implementation in I3SimBatchHandler (based on tfrecords)
class I3SimBatchHandlerFtr:
    @tf.autograph.experimental.do_not_convert
    def __init__(self, sim_handler, process_n_events=None, batch_size=100, n_seq_len_bins=1, remove_pre_pulses=False):
        self.sim_handler = sim_handler
        self.n_events = len(sim_handler.events_meta)
        self.batch_size = batch_size
        if process_n_events is not None:
            if process_n_events > self.n_events:
                print(f"Warning: we only process {self.n_events} events, which is all there is in the file."
                "Set a reasonable value for process_n_events to avoid this warning.")

            else:
                self.n_events = process_n_events

        pulse_data = []
        meta_data = []
        self.n_doms_max = 0
        for i in range(self.n_events):
            meta, pulses = sim_handler.get_event_data(i)
            event_data = sim_handler.get_per_dom_summary_from_sim_data(meta, pulses)
            if remove_pre_pulses:
                sim_handler.replace_early_pulse(event_data, pulses)

            x = event_data[['x', 'y','z','time', 'charge']].to_numpy()
            y = meta[['muon_energy_at_detector', 'q_tot', 'muon_zenith', 'muon_azimuth', 'muon_time',
                      'muon_pos_x', 'muon_pos_y', 'muon_pos_z', 'spline_mpe_zenith',
                      'spline_mpe_azimuth', 'spline_mpe_time', 'spline_mpe_pos_x',
                      'spline_mpe_pos_y', 'spline_mpe_pos_z']].to_numpy()

            pulse_data.append(x)
            meta_data.append(y)
            if x.shape[0] > self.n_doms_max:
                self.n_doms_max = x.shape[0]

        self.n_doms_max += 1
        pulse_data_tf = tf.ragged.constant(pulse_data, ragged_rank=1, dtype=tf.float64)
        meta_data_tf = tf.constant(meta_data, dtype=tf.float64)

        # TF's batch by sequence length magic
        if n_seq_len_bins == 1:
            ds = tf.data.Dataset.from_tensor_slices((pulse_data_tf, meta_data_tf))
            ds = ds.map(lambda x, y: (x, y))
            _element_length_funct = lambda x, y: tf.shape(x)[0]
            ds = ds.bucket_by_sequence_length(
                        element_length_func = _element_length_funct,
                        bucket_boundaries = [self.n_doms_max],
                        bucket_batch_sizes = [self.batch_size, 1],
                        drop_remainder = False,
                        pad_to_bucket_boundary=True
                    )

        else:
            raise NotImplementedError("Parameters for a smart seqlen binning remain to be added."
                    "For now we simply create batches up to max number of DOMs. I.e. choose n_seq_len_bins=1.")

        self.tf_dataset = ds

    def get_batch_iterator(self):
        return iter(self.tf_dataset)


class I3SimBatchHandlerTFRecord:
    @tf.autograph.experimental.do_not_convert
    def __init__(self, infile, batch_size=128, n_features=5, n_labels=14, pad_to_bucket_boundary=True, n_bins=20, bucket_batch_sizes=None, n_doms_max=5170):
        self.tf_dataset = tfrecords_reader_dataset(infile,
                                                    batch_size=batch_size,
                                                    n_features=n_features,
                                                    n_labels=n_labels,
                                                    n_bins=n_bins,
                                                    pad_to_bucket_boundary=pad_to_bucket_boundary,
                                                    bucket_batch_sizes=bucket_batch_sizes,
                                                    n_doms_max = n_doms_max)

    def get_batch_iterator(self):
        return iter(self.tf_dataset)


def parse_tfr_element(element, n_features=5, n_labels=20):
  data = {
      'features': tf.io.FixedLenFeature([], tf.string),
      'labels': tf.io.FixedLenFeature([], tf.string),
    }

  content = tf.io.parse_single_example(element, data)
  labels = content['labels']
  features = content['features']

  feature = tf.io.parse_tensor(features, out_type=tf.float64)
  feature = tf.ensure_shape(feature, (None, n_features))

  label = tf.io.parse_tensor(labels, out_type=tf.float64)
  label = tf.ensure_shape(label, (n_labels,))

  return (feature, label)


def tfrecords_reader_dataset(infile, batch_size, n_features=5, n_labels=20, pad_to_bucket_boundary=True, n_bins=20, bucket_batch_sizes=None, n_doms_max=5170):
    if '*' in infile:
        dataset = tf.data.Dataset.list_files(infile, shuffle=False)
        dataset = tf.data.TFRecordDataset(dataset, compression_type='')

    else:
        dataset = tf.data.TFRecordDataset(infile, compression_type='')

    parse = lambda x: parse_tfr_element(x, n_features=n_features, n_labels=n_labels)
    dataset = dataset.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)

    #n_doms_max = 1000
    #n_bins = 8
    #_element_length_funct = lambda x, y: tf.shape(x)[0]
    #dataset = dataset.bucket_by_sequence_length(
    #        element_length_func = _element_length_funct,
    #        bucket_boundaries = np.logspace(1, np.log10(n_doms_max), n_bins+1).astype(int).tolist(),
    #        bucket_batch_sizes = [batch_size]*(n_bins+2),
    #        drop_remainder = False,
    #        pad_to_bucket_boundary=False,
    #    )

    edges = np.logspace(0.5, np.log10(n_doms_max), n_bins+1).astype(int)
    if bucket_batch_sizes is None:
        #factor = np.median(edges[1:] / edges[:-1])
        factor = 1.3
        scale = np.power(factor, np.arange(n_bins+2)[::-1])
        bucket_batch_sizes = scale * batch_size

    else:
        bucket_batch_sizes = np.array(bucket_batch_sizes)

    #bucket_batch_sizes = 20 * np.ones_like(scale)
    #bucket_batch_sizes = np.clip(bucket_batch_sizes, a_min=1.0, a_max=30.0)
    #print(bucket_batch_sizes)
    #print(edges)

    bucket_batch_sizes = np.clip(bucket_batch_sizes, a_min=1, a_max=None) # batch size at least 1
    bucket_batch_sizes = bucket_batch_sizes.astype(int)

    _element_length_funct = lambda x, y: tf.shape(x)[0]
    dataset = dataset.bucket_by_sequence_length(
            element_length_func = _element_length_funct,
            bucket_boundaries = edges.tolist(),
            bucket_batch_sizes = bucket_batch_sizes.tolist(),
            drop_remainder = False,
            pad_to_bucket_boundary=pad_to_bucket_boundary,
        )

    return dataset.prefetch(tf.data.AUTOTUNE)
