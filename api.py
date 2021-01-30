import flask
from flask import request, jsonify, send_from_directory, json
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from pydub import AudioSegment
# main NN
import pandas as pd
import librosa
import numpy as np
from librosa import feature
import csv
from scipy.fftpack import fft
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import json

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True


# extraction part
list_of_spectral_features = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff
]
list_of_time_domain_features = [
    feature.rms,
    feature.zero_crossing_rate
]


def get_feature_vector(data, sr):
    feat_vect_i = [np.mean(funct(data, sr))
                   for funct in list_of_spectral_features]
    feat_vect_ii = [np.mean(funct(data))
                    for funct in list_of_time_domain_features]
    feature_vector = feat_vect_i + feat_vect_ii
    return feature_vector


sample_features = []

# NN part
# csv file with features revie
csv_file_path = './train_luscinia_corvus.csv'
prepeared_data = pd.read_csv(csv_file_path)

features = [
    'chroma_stft',
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_rolloff',
    'rms',
    'zero_crossing_rate',
]

# Forest Model terain
X = preprocessing.normalize(prepeared_data[features])
y = prepeared_data.name

forest_model = RandomForestRegressor(max_leaf_nodes=80, random_state=1)
forest_model.fit(X, y)


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route('/uploadfile', methods=['GET', 'POST'])
def uploadfile():

    print(request.files)
    if request.method == 'POST':
        raw_file = request.files['audio_data']
        filename = secure_filename(raw_file.filename.replace('.', '-')+'.wav')

        # cut 5sec
        t1 = 500
        t2 = 5500
        new_audio = AudioSegment.from_file(raw_file, 'wav')
        new_audio = new_audio[t1:t2]

        # normalizing
        new_audio = match_target_amplitude(new_audio, -5.0)

        # save result
        new_audio.export(filename, format='wav')
        path = './{}'.format(filename)

        # getting features
        path = './{}'.format(filename)
        data, sr = librosa.load(path, sr=44100)
        feature_vector = get_feature_vector(data, sr)
        sample_features.append(feature_vector)

        print(sample_features)

        val_predictions_test = forest_model.predict(sample_features)

        print(val_predictions_test)
        # forest_prediction = []
        the_bird_prediction = []

        # elif val_predictions_test[0] > 7.5:

        if val_predictions_test[0] < 2.9:
            name = 'luscinia_luscinia'
            name_l = 'luscinia_luscinia'
            url = ''
        elif val_predictions_test[0] > 3:
            name = 'Wrona siwa'
            name_l = '(Corvus corone)'
            url = 'https://upload.wikimedia.org/wikipedia/commons/8/8d/Hooded_Crow_%28Corvus_cornix%29_%2811%29.jpg'
        else:
            name = 'We cant predict the bird'
            name_l = ''
            url = ''

        sample_features.clear()
        print(val_predictions_test, "magic")
        # final_prediction = str(val_predictions_test[0])
        final_prediction = '''	Wrona siwa, wrona (Corvus corone) – gatunek średniego ptaka z rodziny krukowatych (Corvidae), zasadniczo wędrowny, choć duża część osobników jest już osiadła (zwłaszcza populacje miejskie).
		   Występuje w północnej i wschodniej Europie od Półwyspu Apenińskiego i Łaby po Ural. Pierwotnie specjacja następowała w południowo-wschodniej Europie i w cieplejszych strefach Azji. Gnieździ się na rozleglejszym terytorium niż czarnowron. Wyraźne wędrówki i regularne koczowanie podejmują przeważnie młode ptaki (choć tylko po Europie). W Europie Środkowej w pasie o szerokości kilkudziesięciu kilometrów (70–150 km) o rozciągnięciu południkowym wrona występuje wraz z czarnowronem (ten występuje głównie w Europie Zachodniej). To powoduje powstawanie tu mieszanych par i mieszańców z różnym udziałem w upierzeniu barwy czarnej i szarej. Ewolucja nie preferuje żadnego z tych gatunków, a i mieszańce nie wykazują większego dostosowania adaptacyjnego do panujących warunków, toteż ta niewyraźna linia graniczna między nimi pozostaje od lat bez zmian. W innym przypadku bardziej ekspansywna forma rozprzestrzeniłaby się stopniowo na terytorium drugiej z nich. Czasem jednak spotyka się osobniki jednego gatunku w głębi areału zamieszkiwanego przez drugi. Są to głównie osobniki młode, które jeszcze nie założyły własnych gniazd i prowadzą koczowniczy tryb życia. Znajdują w nowych warunkach inaczej upierzonych partnerów i osiadają tam już na stałe.
		   W Polsce średnio liczny ptak lęgowy. Najliczniejsza w górach (dolatuje do 1300 m n.p.m.) i na pogórzu, w dolinach rzek i nad jeziorami. Od lat 30. ubiegłego wieku zaczęła gnieździć się w Warszawie, a później także w innych większych miastach (Poznań – od lat 50., Wrocław, Kraków, Gdańsk – od lat 70.). Dawniej w pobliżu dużych miast zimowała także duża liczba ptaków ze wschodu Europy, obecnie przylatuje ich znacznie mniej, a na zimowych noclegowiskach dominują ptaki z populacji osiadłych. Intensywniejsze przeloty zaznaczają się głównie na wybrzeżu, w głębi lądu nie są już tak wyraźnie widoczne.
		'''

        final_data = {'prediction': final_prediction,
                      'name': name, 'nameL': name_l, 'URL': url}

        print(final_data)

        return jsonify(list([final_data]))

app.run()
