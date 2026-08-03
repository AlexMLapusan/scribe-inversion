import torch
from sentence_transformers import SentenceTransformer

class SentenceTransformerWrapper:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', convert_to_tensor=True, batch_size=100, trust_remote_code=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.convert_to_tensor = convert_to_tensor
        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        self.model.to(self.device)
        self.model.eval()
        # Freeze all the parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = self.model.tokenizer
        self.total_tokens_embedded = 0

    def get_token_count(self):
        """Returns the current total token count."""
        return self.total_tokens_embedded

    def reset_token_count(self):
        """Resets the token counter to 0."""
        self.total_tokens_embedded = 0

    def count_tokens(self, texts):
        """Count tokens for a list of texts without encoding (utility method)."""
        if isinstance(texts, str):
            texts = [texts]
        total = 0
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            total += len(tokens)
        return total

    def _count_and_accumulate_tokens(self, texts):
        """Count tokens in texts and add to the running total."""
        if isinstance(texts, str):
            texts = [texts]
        tokens_in_batch = 0
        for text in texts:
            tokenized = self.tokenizer.encode(text, add_special_tokens=True)
            tokens_in_batch += len(tokenized)
        self.total_tokens_embedded += tokens_in_batch
        return tokens_in_batch

    def get_logits(self, texts, prompt_name=None):
        # Ensure texts is a list for token counting
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = texts

        self._count_and_accumulate_tokens(texts_list)

        with torch.no_grad():
            if prompt_name:
                cls_embeddings = self.model.encode(texts, convert_to_tensor=self.convert_to_tensor,
                                                   batch_size=self.batch_size, prompt_name=prompt_name)
            else:
                cls_embeddings = self.model.encode(texts, convert_to_tensor=self.convert_to_tensor,
                                                   batch_size=self.batch_size)
        return cls_embeddings

    def get_gradual_embeddings(self, texts, chunk_size=1, prompt_name=None):
        embeddings = {}
        chunked_texts = {}
        for text_index, text in enumerate(texts):
            text_words = (text.split(' '))
            chunks = [text_words[0: chunk_size * (j + 1)] for j in range(len(text_words) // chunk_size)]
            embeddings[text_index] = {}
            chunked_texts[text_index] = {}
            # Get the embeddings for each chunk and add them to the embeddings dictionary under the appropriate keys
            for chunk_index, chunk in enumerate(chunks):
                chunk_text = " ".join(chunk)
                chunked_texts[text_index][chunk_index] = chunk_text
                self._count_and_accumulate_tokens([chunk_text])
                with torch.no_grad():
                    if prompt_name:
                        cls_embedding = self.model.encode([chunk_text], convert_to_tensor=self.convert_to_tensor,
                                                          batch_size=self.batch_size, prompt_name=prompt_name)
                    else:
                        cls_embedding = self.model.encode([chunk_text], convert_to_tensor=self.convert_to_tensor,
                                                          batch_size=self.batch_size)
                embeddings[text_index][chunk_index+1] = cls_embedding
        return embeddings, chunked_texts
