import pytest

from app.languages import LANGUAGES, UnsupportedLanguage, detect_language, get_language, matches_language
from app.profanity import filter_profanity


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Привет, как дела?", "ru"),
        ("Сәлеметсіз бе, қалыңыз қалай?", "kk"),
        ("Hello, how are you?", "en"),
        ("12345 !!!", None),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_matches_language():
    assert matches_language("Сәлем, мен жақсымын", LANGUAGES["kk"])
    assert matches_language("Рахмет", LANGUAGES["kk"])  # short, no Kazakh-only letters
    assert not matches_language("Hello there, how can I help you today?", LANGUAGES["kk"])
    assert not matches_language(
        "Это длинный ответ полностью на русском языке без казахских букв", LANGUAGES["kk"]
    )
    assert not matches_language("Привет, чем могу помочь?", LANGUAGES["en"])


@pytest.mark.parametrize("code,expected", [("kz", "kk"), ("RU-ru", "ru"), ("en-US", "en")])
def test_language_aliases(code, expected):
    assert get_language(code).code == expected


def test_unsupported_language():
    with pytest.raises(UnsupportedLanguage):
        get_language("de")


@pytest.mark.parametrize(
    "text",
    [
        "Ну ты и сука",
        "Это пиздец какой-то",
        "Блять, опять дождь",
        "Он меня заебал",
        "What the fuck is this",
        "This is bullshit",
        "Сен қотақсың",
    ],
)
def test_profanity_detected(text):
    result = filter_profanity(text)
    assert result.has_profanity
    assert "*" in result.text


@pytest.mark.parametrize(
    "text",
    [
        "Нужно правильно употреблять слова",
        "Я страхую машину",
        "Голубое небо и свежий хлеб",
        "Сукно хорошего качества",
        "Хулиганы во дворе",
        "Мен сізге көмектесемін",
        "Assessment of the class",
        "Scunthorpe United",
    ],
)
def test_no_false_positives(text):
    result = filter_profanity(text)
    assert not result.has_profanity, result.found
    assert result.text == text


def test_mask_keeps_first_letter_and_length():
    result = filter_profanity("fuck!")
    assert result.text == "f***!"
    assert result.found == ["fuck"]
