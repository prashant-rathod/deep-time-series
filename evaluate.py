from models.darnn.dataset import Dataset
from reportgenerator import ReportGenerator
from data_loaders import m3comp, mg
from models.elmanmodeleval import ElmanModelEval
from models.darnneval import DA_RNN
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def run(models):
    rgenerator = ReportGenerator('test_report')
    for model in models:
        print("Evaluating model - " + model.get_name())

        _generate_mg_report(rgenerator, model)
        _generat_M3_report(rgenerator, model)

    rgenerator.saveReport()

def _generate_mg_report(rgenerator, model):
    X_train, X_test, y_train, y_test = mg.create_test_train_split(mg.mackey_glass())
    rgenerator.generate_report(X_train, X_test, y_train, y_test, model, 'MackeyGlass', model.get_name())

def _generat_M3_report(rgenerator, model):
    dfy = m3comp.load_M3Year()
    for column in dfy:
        series = dfy[column].dropna()
        X_train, X_test, y_train, y_test = m3comp.create_test_train_split(series)
        rgenerator.generate_report(X_train, X_test, y_train, y_test, model, 'M3Year' + column, model.get_name())

    dfq = m3comp.load_M3Quart()
    for column in dfq:
        series = dfq[column].dropna()
        X_train, X_test, y_train, y_test = m3comp.create_test_train_split(series)
        rgenerator.generate_report(X_train, X_test, y_train, y_test, model, 'M3Quart' + column, model.get_name())

    dfm = m3comp.load_M3Month()
    for column in dfm:
        series = dfm[column].dropna()
        X_train, X_test, y_train, y_test = m3comp.create_test_train_split(series)
        rgenerator.generate_report(X_train, X_test, y_train, y_test, model, 'M3Month' + column, model.get_name())

    dfo = m3comp.load_M3Other()
    for column in dfo:
        series = dfo[column].dropna()
        X_train, X_test, y_train, y_test = m3comp.create_test_train_split(series)
        rgenerator.generate_report(X_train, X_test, y_train, y_test, model, 'M3Other' + column, model.get_name())

def _constructDaRNN():
    signal = mg.mackey_glass()
    x = signal[:-1]
    y = signal[1:]
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    dataset = Dataset(x, y, 1)
    darnn = DA_RNN(dataset)
    return darnn

if __name__ == "__main__":
    models = [_constructDaRNN()]
    run(models)
    print("All models have been evaluated")
