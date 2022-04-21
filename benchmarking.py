# 3rd party packages
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# custom modules
from util import *

def exp1(df_data, pretrained_tagger):
    """Only Expand Contractions"""
    # preprocessing
    pp = Preprocessing(df_data)
    df = pp.expand_contractions()

    dp = DatasetPreparator(df, pretrained_tagger)
    df = dp.check_post()
    df = dp.add_bio_ne()
    
    getter = SentenceGetter(df)
    sentences = getter.sentences

    # feature extraction
    print(f"Feature extraction is in progress...")
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    print(f"Feature extraction is complete.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    classes = np.unique(df[["bio_ne"]])
    classes = classes.tolist()
    classes.remove("O")
    new_classes = classes.copy()

    # training
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False
    )
    crf.fit(X_train, y_train)

    # testing
    y_pred = crf.predict(X_test)
    y_test_flat = [item for sublist in y_test for item in sublist ]
    y_pred_flat = [item for sublist in y_pred for item in sublist ]
    print(classification_report(y_test_flat, y_pred_flat, labels = new_classes))

def exp2(df_data, pretrained_tagger):
    """Only Hyphen and Comma Splitting"""
    # preprocessing
    pp = Preprocessing(df_data)
    df = pp.hyphen_comma_splitting()

    dp = DatasetPreparator(df, pretrained_tagger)
    df = dp.check_post()
    df = dp.add_bio_ne()
    
    getter = SentenceGetter(df)
    sentences = getter.sentences

    # feature extraction
    print(f"Feature extraction is in progress...")
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    print(f"Feature extraction is complete.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    classes = np.unique(df[["bio_ne"]])
    classes = classes.tolist()
    classes.remove("O")
    new_classes = classes.copy()

    # training
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False
    )
    crf.fit(X_train, y_train)

    # testing
    y_pred = crf.predict(X_test)
    y_test_flat = [item for sublist in y_test for item in sublist ]
    y_pred_flat = [item for sublist in y_pred for item in sublist ]
    print(classification_report(y_test_flat, y_pred_flat, labels = new_classes))

def exp3(df_data, pretrained_tagger):
    """Only Lowercasing"""
    # preprocessing
    pp = Preprocessing(df_data)
    df = pp.lowercasing()

    dp = DatasetPreparator(df, pretrained_tagger)
    df = dp.check_post()
    df = dp.add_bio_ne()
    
    getter = SentenceGetter(df)
    sentences = getter.sentences

    # feature extraction
    print(f"Feature extraction is in progress...")
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    print(f"Feature extraction is complete.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    classes = np.unique(df[["bio_ne"]])
    classes = classes.tolist()
    classes.remove("O")
    new_classes = classes.copy()

    # training
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False
    )
    crf.fit(X_train, y_train)

    # testing
    y_pred = crf.predict(X_test)
    y_test_flat = [item for sublist in y_test for item in sublist ]
    y_pred_flat = [item for sublist in y_pred for item in sublist ]
    print(classification_report(y_test_flat, y_pred_flat, labels = new_classes))

def exp4(df_data, pretrained_tagger):
    """Only Stemming"""
    # preprocessing
    pp = Preprocessing(df_data)
    df = pp.stemming()

    dp = DatasetPreparator(df, pretrained_tagger)
    df = dp.check_post()
    df = dp.add_bio_ne()
    
    getter = SentenceGetter(df)
    sentences = getter.sentences

    # feature extraction
    print(f"Feature extraction is in progress...")
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    print(f"Feature extraction is complete.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    classes = np.unique(df[["bio_ne"]])
    classes = classes.tolist()
    classes.remove("O")
    new_classes = classes.copy()

    # training
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False
    )
    crf.fit(X_train, y_train)

    # testing
    y_pred = crf.predict(X_test)
    y_test_flat = [item for sublist in y_test for item in sublist ]
    y_pred_flat = [item for sublist in y_pred for item in sublist ]
    print(classification_report(y_test_flat, y_pred_flat, labels = new_classes))

def exp5(df_data, pretrained_tagger):
    """Only Number to Words Conversion"""
    # preprocessing
    pp = Preprocessing(df_data)
    df = pp.number2words()

    dp = DatasetPreparator(df, pretrained_tagger)
    df = dp.check_post()
    df = dp.add_bio_ne()
    
    getter = SentenceGetter(df)
    sentences = getter.sentences

    # feature extraction
    print(f"Feature extraction is in progress...")
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    print(f"Feature extraction is complete.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    classes = np.unique(df[["bio_ne"]])
    classes = classes.tolist()
    classes.remove("O")
    new_classes = classes.copy()

    # training
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False
    )
    crf.fit(X_train, y_train)

    # testing
    y_pred = crf.predict(X_test)
    y_test_flat = [item for sublist in y_test for item in sublist ]
    y_pred_flat = [item for sublist in y_pred for item in sublist ]
    print(classification_report(y_test_flat, y_pred_flat, labels = new_classes))

def exp6(df_data, pretrained_tagger):
    """Only Number to Words Conversion"""
    # preprocessing
    pp = Preprocessing(df_data)
    df = pp.expand_contractions()
    df = pp.hyphen_comma_splitting()
    df = pp.lowercasing()
    df = pp.stemming()
    df = pp.number2words()

    dp = DatasetPreparator(df, pretrained_tagger)
    df = dp.check_post()
    df = dp.add_bio_ne()
    
    getter = SentenceGetter(df)
    sentences = getter.sentences

    # feature extraction
    print(f"Feature extraction is in progress...")
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    print(f"Feature extraction is complete.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    classes = np.unique(df[["bio_ne"]])
    classes = classes.tolist()
    classes.remove("O")
    new_classes = classes.copy()

    # training
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False
    )
    crf.fit(X_train, y_train)

    # testing
    y_pred = crf.predict(X_test)
    y_test_flat = [item for sublist in y_test for item in sublist ]
    y_pred_flat = [item for sublist in y_pred for item in sublist ]
    print(classification_report(y_test_flat, y_pred_flat, labels = new_classes))