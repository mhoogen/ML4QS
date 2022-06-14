##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import numpy as np
import pandas as pd

# This class performs a Fourier transformation on the data to find frequencies that occur
# often and filter noise.
class FourierTransformation:
    
    def __init__(self):
        self.temp_list = []
        self.freqs = None

    # Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
    # the sampling rate expresses
    # the number of samples per second (i.e. Frequency is Hertz of the dataset).
    
    def find_fft_transformation(self, data):
        # Create the transformation, this includes the amplitudes of both the real
        # and imaginary part.
        # print(data.shape)
        transformation = np.fft.rfft(data, len(data))
        # real
        real_ampl = transformation.real
        # max
        max_freq = self.freqs[np.argmax(real_ampl[0:len(real_ampl)])]
        # weigthed
        freq_weigthed = float(np.sum(self.freqs * real_ampl)) / np.sum(real_ampl)

        # pse

        PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
        PSD_pdf = np.divide(PSD, np.sum(PSD))

        # Make sure there are no zeros.
        if np.count_nonzero(PSD_pdf) == PSD_pdf.size:
            pse = -np.sum(np.log(PSD_pdf) * PSD_pdf)
        else:
            pse = 0

        real_ampl = np.insert(real_ampl, 0, max_freq)
        real_ampl = np.insert(real_ampl, 0, freq_weigthed)
        row = np.insert(real_ampl, 0, pse)

        self.temp_list.append(row)

        return 0

    # Get frequencies over a certain window.
    def abstract_frequency(self, data_table, columns, window_size, sampling_rate):
        self.freqs = (sampling_rate * np.fft.rfftfreq(int(window_size))).round(3)

        for col in columns:
            collist = []
            # prepare column names
            collist.append(col + '_max_freq')
            collist.append(col + '_freq_weighted')
            collist.append(col + '_pse')
            
            collist = collist + [col + '_freq_' +
                    str(freq) + '_Hz_ws_' + str(window_size) for freq in self.freqs]
           
            # rolling statistics to calculate frequencies, per window size. 
            # Pandas Rolling method can only return one aggregation value. 
            # Therefore values are not returned but stored in temp class variable 'temp_list'.

            # note to self! Rolling window_size would be nicer and more logical! In older version windowsize is actually 41. (ws + 1)
            data_table[col].rolling(
                window_size + 1).apply(self.find_fft_transformation)

            # Pad the missing rows with nans
            frequencies = np.pad(np.array(self.temp_list), ((window_size, 0), (0, 0)),
                        'constant', constant_values=np.nan)
            # add new freq columns to frame
            
            data_table[collist] = pd.DataFrame(frequencies, index=data_table.index)

            # reset temp-storage array
            del self.temp_list[:]
            

        
        return data_table
