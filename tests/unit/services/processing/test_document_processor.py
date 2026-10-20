from services.processing.document_processor import DocumentProcessor


def test_normalize_english_linebreaks_repairs_split_words():
    result = DocumentProcessor._normalize_english_linebreaks(
        "The pa\nrties shall bear the cost\nof transport."
    )

    assert result == "The parties shall bear the cost of transport."


def test_repealed_article_range_is_preserved_as_repealed():
    assert DocumentProcessor._repealed_by_number(54) is True
    assert DocumentProcessor._repealed_by_number(80) is True


def test_articles_outside_repealed_range_are_not_marked_repealed():
    assert DocumentProcessor._repealed_by_number(53) is False
    assert DocumentProcessor._repealed_by_number(81) is False
