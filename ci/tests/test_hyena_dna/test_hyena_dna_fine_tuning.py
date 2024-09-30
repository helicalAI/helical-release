import pytest
import torch
from helical import HyenaDNA, HyenaDNAConfig, HyenaDNAFineTuningModel

class TestHyenaDNAFineTuning:
    @pytest.fixture(params=["hyenadna-tiny-1k-seqlen", "hyenadna-tiny-1k-seqlen-d256"])
    def hyenaDNA(self, request):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = HyenaDNAConfig(model_name=request.param, device=self.device)
        return HyenaDNA(config)

    @pytest.fixture
    def mock_data(self, hyenaDNA):
        input_sequences = ["AAAA", "CCCC", "TTTT", "ACGT", "ACGN", "BHIK", "ANNT"]
        labels = [0, 0, 0, 0, 0, 0, 0]
        tokenized_sequences = hyenaDNA.process_data(input_sequences)
        return tokenized_sequences, labels
    
    @pytest.fixture
    def hyena_dna_fine_tune(self, hyenaDNA, mock_data):
        input_sequences, labels = mock_data
        model = HyenaDNAFineTuningModel(hyena_model=hyenaDNA, fine_tuning_head="classification", output_size=1)
        model.train(train_input_data=input_sequences, train_labels=labels)
        return model

    def test_output_dimensionality_of_fine_tuned_model(self, hyena_dna_fine_tune, mock_data):
        outputs = hyena_dna_fine_tune.get_outputs(mock_data[0])
        assert outputs.shape == (len(mock_data[0]), 1)