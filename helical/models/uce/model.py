import logging
import numpy as np
from anndata import AnnData
from torch.utils.data import DataLoader
from helical.models.uce.uce_config import UCEConfig
from helical.models.helical import HelicalBaseModel
from helical.models.uce.uce_utils import get_ESM2_embeddings, load_model, process_data, get_gene_embeddings
from accelerate import Accelerator

class UCE(HelicalBaseModel):
    """Universal Cell Embedding Model. This model reads in single-cell RNA-seq data and outputs gene embeddings. 
        This model particularly uses protein-embeddings generated by ESM2. 
        Currently we support human and macaque species but you can add your own species by providing the protein embeddings.

        Example
        -------
        >>> from helical.models import UCE, UCEConfig
        >>> import anndata as ad
        >>> configurer=UCEConfig(batch_size=10)
        >>> uce = UCE(configurer=configurer)
        >>> ann_data = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
        >>> dataset = uce.process_data(ann_data[:100])
        >>> embeddings = uce.get_embeddings(dataset)

        Parameters
        ----------
        configurer : UCEConfig, optional, default = default_configurer
            The model configuration.

        Returns
        -------
        None

        Notes
        -----
        The Universal Cell Embedding Papers has been published on `BioRxiv <https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1>`_ and it is built on top of `SATURN <https://www.nature.com/articles/s41592-024-02191-z>`_ published in Nature.
        """
    default_configurer = UCEConfig()

    def __init__(self, configurer: UCEConfig = default_configurer) -> None:    
        super().__init__()
        self.config = configurer.config
        self.log = logging.getLogger("UCE-Model")

        self.model_dir = self.config['model_path'].parent

        self.embeddings = get_ESM2_embeddings(self.config["token_file_path"], self.config["token_dim"])
        self.model =  load_model(self.config['model_path'], self.config, self.embeddings)
        self.model = self.model.eval()

        if self.config["accelerator"]:
            self.accelerator = Accelerator(project_dir=self.model_dir, cpu=self.config["accelerator"]["cpu"])
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None
        self.log.info(f"Model finished initializing.")

    def process_data(self, data: AnnData, 
                     species: str = "human", 
                     filter_genes_min_cell: int = None, 
                     embedding_model: str = "ESM2" ) -> DataLoader:
        """Processes the data for the Universal Cell Embedding model

        Parameters 
        ----------
        data : AnnData
            The AnnData object containing the data to be processed. 
            The UCE model requires the gene expression data as input and the gene symbols as variable names (i.e. as adata.var_names).
        species: str, optional, default = "human"
            The species of the data.  Currently we support "human" and "macaca_fascicularis" but more embeddings will come soon.
        filter_genes_min_cell: int, default = None
            Filter threshold that defines how many times a gene should occur in all the cells.
        embedding_model: str, optional, default = "ESM2"
            The name of the gene embedding model. The current option is only ESM2.

        Returns
        -------
        DataLoader
            The DataLoader object containing the processed data
        """
        
        files_config = {
            "spec_chrom_csv_path": self.model_dir / "species_chrom.csv",
            "protein_embeddings_dir": self.model_dir / "protein_embeddings/",
            "offset_pkl_path": self.model_dir / "species_offsets.pkl"
        }

        data_loader = process_data(data, 
                              model_config=self.config, 
                              files_config=files_config,
                              species=species,
                              filter_genes_min_cell=filter_genes_min_cell,
                              embedding_model=embedding_model,
                              accelerator=self.accelerator)
        return data_loader

    def get_embeddings(self, dataloader: DataLoader) -> np.array:
        """Gets the gene embeddings from the UCE model

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader object containing the processed data

        Returns
        -------
        np.array
            The gene embeddings in the form of a numpy array
        """
        self.log.info(f"Inference started")
        embeddings = get_gene_embeddings(self.model, dataloader, self.accelerator)
        return embeddings
