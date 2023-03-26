import os
import logging
import torch
import pytorch_lightning as pl
import warnings
warnings.filterwarnings('ignore')
import torch
import pytorch_lightning as pl
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from functools import partial
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.data_load.datasets import inference_data_loader
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm
import argparse

def parse_handle():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frac', help="Pretrain size", type=float, default=1.)
    parser.add_argument('--hidden_size', help="Encoder size", type=int, default=64)

    return parser

parser = parse_handle()
args = parser.parse_args()


if not os.path.exists('data_sber/transactions_train.csv'):
    os.system('mkdir -p data_sber')
    os.system('curl -OL https://storage.yandexcloud.net/datasouls-competitions/sirius/data.zip')
    os.system("unzip -j -o data.zip '*.csv' -d data_sber")
    os.system('mv data.zip data_sber/')

data_path = 'data_sber/'

source_data = pd.read_csv(os.path.join(data_path, 'transactions_train.csv'))

initial_test = pd.read_csv(os.path.join(data_path, 'transactions_test.csv'))

whole_dataset = pd.concat([source_data, initial_test])

from ptls.preprocessing import PandasDataPreprocessor

preprocessor = PandasDataPreprocessor(
    col_id='client_id',
    col_event_time='trans_date',
    event_time_transformation='none',
    cols_category=['small_group'],
    cols_numerical=['amount_rur'],
    return_records=True,
)

os.system('time')
whole_dataset = preprocessor.fit_transform(whole_dataset.sample(frac=args.frac, random_state=42))

with open('preprocessor.p', 'wb') as f:
    pickle.dump(preprocessor, f)

source_data = preprocessor.fit_transform(source_data)
dataset = sorted(source_data, key=lambda x: x['client_id'])
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

trx_encoder_params = dict(
    embeddings_noise=0.003,
    numeric_values={'amount_rur': 'identity'},
    embeddings={
        'trans_date': {'in': 800, 'out': 16},
        'small_group': {'in': 250, 'out': 16},
    },
)

seq_encoder = RnnSeqEncoder(
    trx_encoder=TrxEncoder(**trx_encoder_params),
    hidden_size=args.hidden_size,
    type='gru',
)

model = CoLESModule(
    seq_encoder=seq_encoder,
    optimizer_partial=partial(torch.optim.Adam, lr=0.001),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9),
)

train_dl = PtlsDataModule(
    train_data=ColesDataset(
        MemoryMapDataset(
            data=whole_dataset,
            i_filters=[
                SeqLenFilter(min_seq_len=25),
            ],
        ),
        splitter=SampleSlices(
            split_count=5,
            cnt_min=25,
            cnt_max=200,
        ),
    ),
    train_num_workers=16,
    train_batch_size=256,
)

trainer = pl.Trainer(
    max_epochs=15,
    gpus=1 if torch.cuda.is_available() else 0,
    enable_progress_bar=False,
)

os.system('time')
print(f'logger.version = {trainer.logger.version}')
trainer.fit(model, train_dl)
print(trainer.logged_metrics)

torch.save(seq_encoder.state_dict(), "coles-emb.pt")

# embedding inference

train_dl = inference_data_loader(train, num_workers=0, batch_size=256)
train_embeds = torch.vstack(trainer.predict(model, train_dl, ))

test_dl = inference_data_loader(test, num_workers=0, batch_size=256)
test_embeds = torch.vstack(trainer.predict(model, test_dl))

# join target and embeddings

df_target = pd.read_csv(os.path.join(data_path, 'train_target.csv'))
df_target = df_target.set_index('client_id')
df_target.rename(columns={"bins": "target"}, inplace=True)

train_df = pd.DataFrame(data=train_embeds, columns=[f'embed_{i}' for i in range(train_embeds.shape[1])])
train_df['client_id'] = [x['client_id'] for x in train]
train_df = train_df.merge(df_target, how='left', on='client_id')

test_df = pd.DataFrame(data=test_embeds, columns=[f'embed_{i}' for i in range(test_embeds.shape[1])])
test_df['client_id'] = [x['client_id'] for x in test]
test_df = test_df.merge(df_target, how='left', on='client_id')

embed_columns = [x for x in train_df.columns if x.startswith('embed')]
x_train, y_train = train_df[embed_columns], train_df['target']
x_test, y_test = test_df[embed_columns], test_df['target']

models = []
scores = []

clf_rf = RandomForestClassifier()
clf_rf.fit(x_train, y_train)
scores.append(clf_rf.score(x_test, y_test))
models.append(clf_rf.__class__.__name__)

clf_lgbm = lightgbm.LGBMClassifier(
    max_depth=6,
    learning_rate=0.02,
    n_estimators=500,
    objective = 'binary',
    subsample= 0.5,
    subsample_freq= 1,
    feature_fraction= 0.75,
    lambda_l1= 1,
    lambda_l2= 1,
    min_data_in_leaf= 50,
    random_state= 42,
    n_jobs= 8
)
clf_lgbm = clf_lgbm.fit(x_train, y_train)
scores.append(clf_lgbm.score(x_test, y_test))
models.append(clf_lgbm.__class__.__name__)

clf_knn =  KNeighborsClassifier(4)
clf_knn.fit(x_train, y_train)
scores.append(clf_knn.score(x_test, y_test))
models.append(clf_knn.__class__.__name__)

clf_dt = DecisionTreeClassifier(max_depth=2)
clf_dt.fit(x_train, y_train)
scores.append(clf_dt.score(x_test, y_test))
models.append(clf_dt.__class__.__name__)

clf_mlp = MLPClassifier([100, 200, 100], alpha=1, max_iter=1000)
clf_mlp.fit(x_train, y_train)
scores.append(clf_mlp.score(x_test, y_test))
models.append(clf_mlp.__class__.__name__)

clf_nb = GaussianNB()
clf_nb.fit(x_train, y_train)
scores.append(clf_nb.score(x_test, y_test))
models.append(clf_nb.__class__.__name__)

df = pd.DataFrame([
    args.hidden_size * 7,
    args.frac * 7,
    models,
    scores
], columns=['a', 'b', 'c'])
df.to_csv(f'sberbank_{args.hidden_size}_{args.frac}.csv')