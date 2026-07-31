from bpe_tokenizer import Tokenizer
import pytest
import tiktoken

@pytest.fixture(scope="module")
def my_tokenizer():
    return Tokenizer()

@pytest.fixture(scope="module")
def reference_tokenizer():
    return tiktoken.get_encoding("gpt2")


def test_empty_string(my_tokenizer):
    assert my_tokenizer.encode("") == []
    assert my_tokenizer.decode([]) == ""


test_cases = [
    "hello world",
    "1234567890",
    "0",
    "00",
    "00000000",
    "today\n ",
    "today\n \n",
    "Complex word: Superfluous, Arachnophobia, etc.",
    "   leading and trailing spaces   ",
    "Special punctuation: !@#$%^&*()_+-=[]{}|;:',.<>/?",
    "Multi-line\ntext\nwith\r\nvarious\rnewlines.",
]

@pytest.mark.parametrize("sentence", test_cases)
def test_tokenization_consistency(my_tokenizer, reference_tokenizer, sentence):
    my_tokens = my_tokenizer.encode(sentence)
    expected_tokens = reference_tokenizer.encode(sentence)

    assert my_tokens == expected_tokens, f"Token mismatch for sentence: {repr(sentence)}"

@pytest.mark.parametrize("sentence", test_cases)
def test_decoding_consistency(my_tokenizer, sentence):
    my_tokens = my_tokenizer.encode(sentence)
    my_decoded_text = my_tokenizer.decode(my_tokens)

    assert my_decoded_text == sentence, f"Decoded text mismatch for sentence: {repr(sentence)}"


def test_repeated_character_tokenization(my_tokenizer, reference_tokenizer):
    for test_char in ["0", "a", " ", "\n"]:
        for test_length in [1, 2, 3, 5, 10]:
            test_string = test_char * test_length
            my_tokens = my_tokenizer.encode(test_string)
            expected_tokens = reference_tokenizer.encode(test_string)

            assert my_tokens == expected_tokens, f"Token mismatch for repeated character: {repr(test_string)}"

def test_invalid_token_handling(my_tokenizer):
    
    with pytest.raises(ValueError):
        my_tokenizer.decode([987654321])