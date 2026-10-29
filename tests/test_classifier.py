import pytest
from app import get_models, LABEL_MAP, CATEGORY_DESCRIPTIONS


def test_models_loader():
    models = get_models()
    assert models is not None
    assert "vectorizer" in models
    assert "nb_model" in models
    assert "lr_model" in models
    assert models["vectorizer"] is not None


def test_category_metadata_completeness():
    assert len(LABEL_MAP) == 5
    for cat_id, cat_name in LABEL_MAP.items():
        assert cat_name in CATEGORY_DESCRIPTIONS
        desc = CATEGORY_DESCRIPTIONS[cat_name]
        assert len(desc) > 10


def test_feature_vectorizer_vocabulary():
    models = get_models()
    vec = models["vectorizer"]
    vocab = vec.vocabulary_
    assert len(vocab) > 5000
    # Core medical vocabulary must exist
    assert "patient" in vocab
    assert "disease" in vocab
    assert "cardiac" in vocab
    assert "therapy" in vocab
