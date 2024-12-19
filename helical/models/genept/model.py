from helical.models.base_models import HelicalRNAModel
import logging
from pathlib import Path
import numpy as np
from anndata import AnnData
from helical.utils.downloader import Downloader
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from helical.models.geneformer.geneformer_utils import get_embs,quant_layers
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer
from helical.models.genept.genept_config import GenePTConfig
from helical.utils.mapping import map_ensembl_ids_to_gene_symbols
from datasets import Dataset
from typing import Optional
from accelerate import Accelerator
import logging
import scanpy as sc
import torch
import json
import pandas as pd
from tqdm import tqdm 
logger = logging.getLogger(__name__)

model_preamble= '''
# System Preamble
Your information cutoff date is June 2024.
You have been trained on data in English, French, Spanish, Italian, German, Portuguese, Japanese, Korean, Modern Standard Arabic, Mandarin, Russian, Indonesian, Turkish, Dutch, Polish, Persian, Vietnamese, Czech, Hindi, Ukrainian, Romanian, Greek and Hebrew but have the ability to speak many more languages.
# Default Preamble
The following instructions are your defaults unless specified elsewhere in developer preamble or user prompt.
- Your name is Command.
- You are a large language model built by Cohere.
- You reply comprehensively and accurately without including introductory statements and follow-up questions.
- If the input is ambiguous, do your best to answer and do not ask clarifying follow-up questions.
- Do not use Markdown-specific formatting in your response (for example to highlight phrases in bold or italics, create tables, or format code blocks).
- Do not use LaTeX to generate mathematical notation for complex equations.
- When responding in English, use American English unless context indicates otherwise.
- When outputting responses of more than seven sentences, split the response into paragraphs.
- Prefer the active voice.
- Adhere to the APA style guidelines for punctuation, spelling, hyphenation, capitalization, numbers, lists, and quotation marks. Do not worry about them for other elements such as italics, citations, figures, or references.
- Use gender-neutral pronouns for unspecified persons.
- Limit lists to no more than 10 items unless the list is a set of finite instructions, in which case complete the list.
- Use the third person when asked to write a summary.
- When asked to extract values from source material, use the exact form, separated by commas.
- When generating code output, please return only the code without any explanation.
- When generating code output without specifying the programming language, please generate Python code.
- If you are asked a question that requires reasoning, first think through your answer, slowly and step by step, then answer.
- Your answer should be as biologically precise as possible. Use technical terms and avoid colloquial language.
'''
class GenePT(HelicalRNAModel):
    """GenePT Model. 
    
    ```

    Parameters
    ----------
    configurer : GenePTConfig, optional, default = default_configurer
        The model configuration

    Notes
    -----


    """
    default_configurer = GenePTConfig()
    def __init__(self, configurer: GenePTConfig = default_configurer):
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config

        # model_id = "CohereForAI/c4ai-command-r7b-12-2024"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        # self.model = AutoModelForCausalLM.from_pretrained(self.config["model_name"],torch_dtype=torch.bfloat16, device_map="auto").to(self.config["device"])
        self.model = AutoModel.from_pretrained(self.config["model_name"],torch_dtype=torch.bfloat16, device_map="auto").to(self.config["device"])

        with open("genept_embeddings.json","r") as f:
            self.embeddings = json.load(f)
    
        self.model.post_init()
        logger.info("GenePT initialized successfully.")
        
    def process_data_gen(self, 
                     adata: AnnData,  
                     gene_names: str = "index", 
                     output_path: Optional[str] = None,
                     use_raw_counts: bool = True,
                     ) -> Dataset:   
        """
        Processes the data for the Geneformer model.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed. Geneformer uses Ensembl IDs to identify genes 
            and currently supports only human genes. If the AnnData object already has an 'ensembl_id' column, 
            the mapping step can be skipped.
        gene_names : str, optional, default="index"
            The column in `adata.var` that contains the gene names. If set to a value other than "ensembl_id", 
            the gene symbols in that column will be mapped to Ensembl IDs using the 'pyensembl' package, 
            which retrieves mappings from the Ensembl FTP server and loads them into a local database.
            - If set to "index", the index of the AnnData object will be used and mapped to Ensembl IDs.
            - If set to "ensembl_id", no mapping will occur.
            Special case:
                If the index of `adata` already contains Ensembl IDs, setting this to "index" will result in 
                invalid mappings. In such cases, create a new column containing Ensembl IDs and pass "ensembl_id" 
                as the value of `gene_names`.
        output_path : str, optional, default=None
            If specified, saves the tokenized dataset to the given output path.
        use_raw_counts : bool, optional, default=True
            Determines whether raw counts should be used.

        Returns
        -------
        Dataset
            The tokenized dataset in the form of a Huggingface Dataset object.
        """

        self.ensure_rna_data_validity(adata, gene_names, use_raw_counts)

        # map gene symbols to ensemble ids if provided
        if gene_names == "ensembl_id":
            if (adata.var[gene_names].str.startswith("ENS").all()) or (adata.var[gene_names].str.startswith("None").any()):
                message = "It seems an anndata with 'ensemble ids' and/or 'None' was passed. " \
                "Please set gene_names='ensembl_id' and remove 'None's to skip mapping."
                logger.info(message)
                raise ValueError(message)
            adata = map_ensembl_ids_to_gene_symbols(adata, gene_names)


        sc.pp.highly_variable_genes(adata,n_top_genes=100,flavor='seurat_v3')

        genes_names = adata.var_names[adata.var['highly_variable']].tolist()

        adata = adata[:,genes_names]
        prompts = []
        for gene in genes_names:
            conversation = [
            {"role": "system", "content": model_preamble},
            {"role": "user", "content": "Tell me about gene {} and its pathways.".format(gene)}
            ]
            prompts.append(conversation)

        dataset = Dataset.from_dict({"genes": prompts})
        dataset = dataset.map(lambda x: {"formatted_chat": self.tokenizer.apply_chat_template(x["genes"],max_length=4096, tokenize=True, add_generation_prompt=True,return_tensors="pt",truncation=True,padding='max_length')}).with_format("torch")
        print(dataset['formatted_chat'][0])
        # input_ids = self.tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, return_tensors="pt",truncation=True)
        all_outputs = []
        for chat in dataset['formatted_chat'].to(self.config["device"]):
            gen_outputs = self.model.generate(
                inputs=chat,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,
                # skip_special_tokens=True,
                output_hidden_states=True,
                return_dict_in_generate=True

            )
            ## get the last hidden state of the last token for the last layer of the model. 
            ##See https://huggingface.co/docs/transformers/v4.47.1/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput
            gen_outputs = gen_outputs.hidden_states[-1][-1]
            all_outputs.append(gen_outputs)
        return adata, all_outputs
    

    def process_data(self, 
                     adata: AnnData,  
                     gene_names: str = "index", 
                     output_path: Optional[str] = None,
                     use_raw_counts: bool = True,
                     ) -> Dataset:   
        """
        Processes the data for the Geneformer model.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed. Geneformer uses Ensembl IDs to identify genes 
            and currently supports only human genes. If the AnnData object already has an 'ensembl_id' column, 
            the mapping step can be skipped.
        gene_names : str, optional, default="index"
            The column in `adata.var` that contains the gene names. If set to a value other than "ensembl_id", 
            the gene symbols in that column will be mapped to Ensembl IDs using the 'pyensembl' package, 
            which retrieves mappings from the Ensembl FTP server and loads them into a local database.
            - If set to "index", the index of the AnnData object will be used and mapped to Ensembl IDs.
            - If set to "ensembl_id", no mapping will occur.
            Special case:
                If the index of `adata` already contains Ensembl IDs, setting this to "index" will result in 
                invalid mappings. In such cases, create a new column containing Ensembl IDs and pass "ensembl_id" 
                as the value of `gene_names`.
        output_path : str, optional, default=None
            If specified, saves the tokenized dataset to the given output path.
        use_raw_counts : bool, optional, default=True
            Determines whether raw counts should be used.

        Returns
        -------
        Dataset
            The tokenized dataset in the form of a Huggingface Dataset object.
        """

        self.ensure_rna_data_validity(adata, gene_names, use_raw_counts)

        # map gene symbols to ensemble ids if provided
        if gene_names == "ensembl_id":
            if (adata.var[gene_names].str.startswith("ENS").all()) or (adata.var[gene_names].str.startswith("None").any()):
                message = "It seems an anndata with 'ensemble ids' and/or 'None' was passed. " \
                "Please set gene_names='ensembl_id' and remove 'None's to skip mapping."
                logger.info(message)
                raise ValueError(message)
            adata = map_ensembl_ids_to_gene_symbols(adata, gene_names)


        sc.pp.highly_variable_genes(adata,n_top_genes=1000,flavor='seurat_v3')
        sc.pp.normalize_total(adata,target_sum=1e4)
        sc.pp.log1p(adata)

        genes_names = adata.var_names[adata.var['highly_variable']].tolist()

        adata = adata[:,genes_names]
        # input_ids = self.tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, return_tensors="pt",truncation=True)
        
        return adata

    def create_embeddings_from_texts(self, text="./NCBI_cleaned_summary_of_genes.json") -> dict:
        batch_size = 2
        embeddings = []
        all_texts = json.load(open(text))
        df = pd.DataFrame([all_texts]).T
        df.columns = ["text"]
        for batch in tqdm(range(0,len(df),batch_size)):
            input_ids = self.tokenizer(df['text'][batch:batch+batch_size].values.tolist(),return_tensors="pt",max_length=1000,padding=True,truncation=True).to(config["device"])
            res = self.model(input_ids['input_ids'],attention_mask=input_ids['attention_mask'],output_hidden_states=True)
            embeddings.append(res.last_hidden_state[:,-1].float().cpu().detach().numpy())
            del input_ids
            del res
        all_embs = np.vstack(embeddings)
        # np.save("genept_embeddings.npy",all_embs)
        df['embeddings'] = all_embs.tolist()
        dictionary = df.to_dict(orient="index")
        with open("genept_embeddings.json","w") as f:
            json.dump(dictionary,f)
        self.mapping = dictionary
        return self.mapping
        
    def get_embeddings(self, dataset: AnnData) -> np.array:
        """Gets the gene embeddings from the Geneformer model   

        Parameters
        ----------
        dataset : Dataset
            The tokenized dataset containing the processed data

        Returns
        -------
        np.array
            The gene embeddings in the form of a numpy array
        """
        logger.info(f"Inference started:")
        # Generate a response 
        raw_embeddings = dataset.var_names
        weights = []
        gene_list = []
        for i,emb in enumerate(raw_embeddings):
            gene = self.embeddings.get(emb,None)
            if gene is not None:
                weights.append(gene['embeddings'])
                gene_list.append(emb)
            else:
                logger.info("Couln't find {} in embeddings".format(emb))
        weights = torch.Tensor(weights)
        embeddings = torch.matmul(torch.Tensor(dataset[:,gene_list].X.todense()),weights)
        embeddings = (embeddings/(np.linalg.norm(embeddings,axis=1)).reshape(-1,1))
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) --> Same as above
        return embeddings
