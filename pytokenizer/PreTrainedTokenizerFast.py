from collections import defaultdict
from tokenizer.PreTrainedTokenizerBase import PreTrainedTokenizerBase
import os
from shutil import copyfile
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from tokenizers import Encoding as EncodingFast
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast

from tokenizer.SpecialTokensMixin import *

VOCAB_FILES_NAMES = {
    "vocab_file": "tokenizer.model",
    "tokenizer_file": "tokenizer.json",
}

class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = None

    def __init__(self, *args, **kwargs):
        tokenizer_object = kwargs.pop("tokenizer_object", None)
        slow_tokenizer = kwargs.pop("__slow_tokenizer", None)
        gguf_file = kwargs.pop("gguf_file", None)
        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)
        from_slow = kwargs.pop("from_slow", False)
        added_tokens_decoder = kwargs.pop("added_tokens_decoder", {})

        if from_slow and slow_tokenizer is None and self.slow_tokenizer_class is None:
            raise ValueError(
                "Cannot instantiate this tokenizer from a slow version. If it's based on sentencepiece, make sure you "
                "have sentencepiece installed."
            )

        if tokenizer_object is not None:
            fast_tokenizer = copy.deepcopy(tokenizer_object)
        elif fast_tokenizer_file is not None and not from_slow:
            # We have a serialization from tokenizers which let us directly build the backend
            fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
        # elif slow_tokenizer:
        #     # We need to convert a slow tokenizer to build the backend
        #     fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        # elif gguf_file is not None:
        #     # We need to convert a slow tokenizer to build the backend
        #     gguf_param = load_gguf_checkpoint(kwargs.get("vocab_file"))
        #     architecture = gguf_param["config"]["model_type"]
        #     tokenizer_dict = gguf_param["tokenizer"]
        #     tokenizer_config = gguf_param["tokenizer_config"]
        #     fast_tokenizer, additional_kwargs = convert_gguf_tokenizer(architecture, tokenizer_dict)
        #     kwargs.update(tokenizer_config)
        #     if len(additional_kwargs) > 0:
        #         kwargs.update(additional_kwargs)
        # elif self.slow_tokenizer_class is not None and slow_tokenizer is not False:
        #     # We need to create and convert a slow tokenizer to build the backend
        #     slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)
        #     fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
        # elif not slow_tokenizer:
        #     # We tried loading a slow_tokenizer with spm and failed, try to load with tiktoken
        #     self.vocab_file = kwargs.get("vocab_file", None)
        #     self.additional_special_tokens = kwargs.get("additional_special_tokens", [])
        #     fast_tokenizer = convert_slow_tokenizer(self, from_tiktoken=True)
        #     slow_tokenizer = None
        else:
            raise ValueError(
                "Couldn't instantiate the backend tokenizer from one of: \n"
                "(1) a `tokenizers` library serialization file, \n"
                "(2) a slow tokenizer instance to convert or \n"
                "(3) an equivalent slow tokenizer class to instantiate and convert. \n"
                "You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one."
            )

        self._tokenizer: TokenizerFast = fast_tokenizer

        if slow_tokenizer is not None:
            kwargs.update(slow_tokenizer.init_kwargs)

        self._decode_use_source_tokenizer = False

        _truncation = self._tokenizer.truncation

        if _truncation is not None:
            self._tokenizer.enable_truncation(**_truncation)
            kwargs.setdefault("max_length", _truncation["max_length"])
            kwargs.setdefault("truncation_side", _truncation["direction"])
            kwargs.setdefault("stride", _truncation["stride"])
            kwargs.setdefault("truncation_strategy", _truncation["strategy"])
        else:
            self._tokenizer.no_truncation()

        _padding = self._tokenizer.padding
        if _padding is not None:
            self._tokenizer.enable_padding(**_padding)
            kwargs.setdefault("pad_token", _padding["pad_token"])
            kwargs.setdefault("pad_token_type_id", _padding["pad_type_id"])
            kwargs.setdefault("padding_side", _padding["direction"])
            kwargs.setdefault("max_length", _padding["length"])
            kwargs.setdefault("pad_to_multiple_of", _padding["pad_to_multiple_of"])

        # We call this after having initialized the backend tokenizer because we update it.
        super().__init__(**kwargs)
        self._tokenizer.encode_special_tokens = self.split_special_tokens

        added_tokens_decoder_hash = {hash(repr(token)) for token in self.added_tokens_decoder}
        tokens_to_add = [
            token
            for index, token in sorted(added_tokens_decoder.items(), key=lambda x: x[0])
            if hash(repr(token)) not in added_tokens_decoder_hash
        ]
        encoder = list(self.added_tokens_encoder.keys()) + [str(token) for token in tokens_to_add]
        # if some of the special tokens are strings, we check if we don't already have a token
        tokens_to_add += [
            token for token in self.all_special_tokens_extended if token not in encoder and token not in tokens_to_add
        ]

        if len(tokens_to_add) > 0:
            tokens = []
            special_tokens = self.all_special_tokens
            for token in tokens_to_add:
                is_special = (
                    (token.special or str(token) in special_tokens)
                    if isinstance(token, AddedToken)
                    else str(token) in special_tokens
                )
                if isinstance(token, str):
                    token = AddedToken(token, special=is_special)
                else:
                    token.special = is_special
                tokens.append(token)
            if tokens:
                self.add_tokens(tokens)

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        `bool`: Whether or not the slow tokenizer can be saved. Usually for sentencepiece based slow tokenizer, this
        can only be `True` if the original `"sentencepiece.model"` was not deleted.
        """
        return True

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        return self.get_vocab()

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return self._tokenizer.get_added_tokens_decoder()

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> TokenizerFast:
        """
        `tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        """
        `tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
        """
        return self._tokenizer.decoder

    def _convert_encoding(
        self,
        encoding: EncodingFast,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[EncodingFast]]:
        """
        Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list
        of encodings, take care of building a batch from overflowing tokens.

        Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
        lists (overflows) of lists (tokens).

        Output shape: (overflows, sequence length)
        """
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        encoding_dict = defaultdict(list)
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)

            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
            if return_length:
                encoding_dict["length"].append(len(e.ids))

        return encoding_dict, encodings

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        return [self._convert_token_to_id_with_added_voc(token) for token in tokens]

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)

        return self._tokenizer.add_tokens(new_tokens)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        return self._tokenizer.num_special_tokens_to_add(pair)

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        return self.encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, **kwargs).tokens()

    def set_truncation_and_padding(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: int,
        stride: int,
        pad_to_multiple_of: Optional[int],
        padding_side: Optional[bool],
    ):
        """
        Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
        library) and restore the tokenizer settings afterwards.

        The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
        padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
        section.

        Args:
            padding_strategy ([`~utils.PaddingStrategy`]):
                The kind of padding that will be applied to the input
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`]):
                The kind of truncation that will be applied to the input
            max_length (`int`):
                The maximum size of a sequence.
            stride (`int`):
                The stride to use when handling overflow.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            padding_side (`str`, *optional*):
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
        """
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            if _truncation is not None:
                self._tokenizer.no_truncation()
        else:
            target = {
                "max_length": max_length,
                "stride": stride,
                "strategy": truncation_strategy.value,
                "direction": self.truncation_side,
            }

            # _truncation might contain more keys that the target `transformers`
            # supports. Use only the target keys to trigger `enable_truncation`.
            # This should enable this code to works on various `tokenizers`
            # targets.
            if _truncation is None:
                current = None
            else:
                current = {k: _truncation.get(k, None) for k in target}

            if current != target:
                self._tokenizer.enable_truncation(**target)

        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            if _padding is not None:
                self._tokenizer.no_padding()
        else:
            length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            target = {
                "length": length,
                "direction": padding_side if padding_side is not None else self.padding_side,
                "pad_id": self.pad_token_id,
                "pad_token": self.pad_token,
                "pad_type_id": self.pad_token_type_id,
                "pad_to_multiple_of": pad_to_multiple_of,
            }
            if _padding != target:
                self._tokenizer.enable_padding(**target)

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
    ) -> BatchEncoding:
        if not isinstance(batch_text_or_text_pairs, (tuple, list)):
            raise TypeError(
                f"batch_text_or_text_pairs has to be a list or a tuple (got {type(batch_text_or_text_pairs)})"
            )

        # Set the truncation and padding strategy and restore the initial configuration
        self.set_truncation_and_padding(
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
        )

        if self._tokenizer.encode_special_tokens != split_special_tokens:
            self._tokenizer.encode_special_tokens = split_special_tokens

        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )

        tokens_and_encodings = [
            self._convert_encoding(
                encoding=encoding,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
            )
            for encoding in encodings
        ]

        # Convert the output to have dict[list] from list[dict] and remove the additional overflows dimension
        # From (variable) shape (batch, overflows, sequence length) to ~ (batch * overflows, sequence length)
        # (we say ~ because the number of overflow varies with the example in the batch)
        #
        # To match each overflowing sample with the original sample in the batch
        # we add an overflow_to_sample_mapping array (see below)
        sanitized_tokens = {}
        for key in tokens_and_encodings[0][0].keys():
            stack = [e for item, _ in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack
        sanitized_encodings = [e for _, item in tokens_and_encodings for e in item]

        # If returning overflowing tokens, we need to return a mapping
        # from the batch idx to the original sample
        if return_overflowing_tokens:
            overflow_to_sample_mapping = []
            for i, (toks, _) in enumerate(tokens_and_encodings):
                overflow_to_sample_mapping += [i] * len(toks["input_ids"])
            sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping

        for input_ids in sanitized_tokens["input_ids"]:
            self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
        return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[bool] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_output = self._batch_encode_plus(
            batched_input,
            is_split_into_words=is_split_into_words,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

        # Return tensor is None, then we can remove the leading batch axis
        # Overflowing tokens are returned as a batch of output so we keep them in this case
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        return batched_output

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.backend_tokenizer.decoder.decode(tokens)

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif isinstance(token_ids, list) and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text
