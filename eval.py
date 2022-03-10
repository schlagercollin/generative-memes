import torch
import enum
from captionmodel import RefinedLanguageModel, EncoderCNN, DecoderRNN
from dataset import get_meme_caption_dataset
from inference import generate_caption_v2_beam_search, gt_vec_to_text, pred_vec_to_text
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from settings import refined_model_vocab_embed_size, refined_model_decoder_hidden_size, refined_model_decoder_num_layers, refined_model_encoder_embed_size

def get_refined_language_model(params_loc, vocab_size):
    model = RefinedLanguageModel(
            vocab_embed_size=refined_model_vocab_embed_size,
            decoder_hidden_size=refined_model_decoder_hidden_size,
            decoder_num_layers=refined_model_decoder_num_layers,
            vocab_size=vocab_size,
            encoder_embed_size=refined_model_encoder_embed_size
        ).to(device)

    model.load_state_dict((torch.load(params_loc, map_location=torch.device(device))))

    return (model,)

def get_baseline_language_model(params_encoder_loc, params_decoder_loc, vocab_size):
    encoder = EncoderCNN()
    decoder = DecoderRNN(
        embed_size=1024, 
        hidden_size=1024, 
        vocab_size=vocab_size
    )

    encoder.load_state_dict((torch.load(params_encoder_loc, map_location=torch.device(device))))
    decoder.load_state_dict((torch.load(params_decoder_loc, map_location=torch.device(device))))

    return (encoder, decoder)

def eval_model(model_name, model, data_loader, dataset, beam_search_temperature=None):

    num_sentences_to_test = 300

    bleu_1_sum = 0
    bleu_2_sum = 0
    bleu_3_sum = 0
    bleu_4_sum = 0
    jaccard_similarity_sum = 0
    
    for _ in tqdm(range(num_sentences_to_test)):
        image, caption_vec = next(data_loader)
        caption_vec = caption_vec.to(device)
        image = image.to(device)

        

        if len(model) == 1:
            # Refined Language Model (unified architecture)

            if beam_search_temperature is None:
                beam_search_temperature = 7 if 'adversarial' in model_name else 1.5

            caption, _ = generate_caption_v2_beam_search(
                image,
                model[0],
                dataset,
                device,
                length_to_generate=10,
                beam_search_temperature=beam_search_temperature,
                branch_factor=1,
                silent=True
            )
            caption = caption.split()

        else:
            # Baseline Language Model (separate architecture)
            encoder, decoder = model
            output_preds = decoder(encoder(image))
            caption = gt_vec_to_text(output_preds, dataset)

        gt_caption = gt_vec_to_text(caption_vec, dataset)

        bleu_1_sum += sentence_bleu(gt_caption, caption, weights=(1, 0, 0, 0))
        bleu_2_sum += sentence_bleu(gt_caption, caption, weights=(0, 1, 0, 0))
        bleu_3_sum += sentence_bleu(gt_caption, caption, weights=(0, 0, 1, 0))
        bleu_4_sum += sentence_bleu(gt_caption, caption, weights=(0, 0, 0, 1))
        jaccard_similarity_sum += len(set(caption).intersection(set(gt_caption[0]))) / len(set(caption).union(set(gt_caption[0])))

    return {
        'bleu-1': bleu_1_sum / num_sentences_to_test,
        'bleu-2': bleu_2_sum / num_sentences_to_test,
        'bleu-3': bleu_3_sum / num_sentences_to_test,
        'bleu-4': bleu_4_sum / num_sentences_to_test,
        'jaccard-similarity': jaccard_similarity_sum / num_sentences_to_test
    }

def eval():

    dataset = get_meme_caption_dataset()
    vocab_size = len(dataset.itos)

    data_loader = iter(torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1
    ))

    model_name_to_model_ckpts = {
        # 'final_supervised': get_refined_language_model('./caption-model-v2-ckpts/v3-epoch-0.ckpt', vocab_size),
        'final_adversarial': get_refined_language_model('./caption-model-adversarial/generator_epoch_1_iter_6000.ckpt', vocab_size),
        # 'baseline': get_baseline_language_model('./caption-model-ckpts/encoder-440.pth', './caption-model-ckpts/decoder-440.pth', vocab_size)
    }

    print("All models initialised")

    temperatures_to_test = [1.0, 1.5, 2.0, 5.0, 7.0, 10.0]
    
    for model_name, model in model_name_to_model_ckpts.items():
        for temp in temperatures_to_test:
            results = eval_model(model_name, model, data_loader, dataset, beam_search_temperature=temp)
            print('Model: ', model_name)
            print('Temperature: ', temp)
            print(results)


if __name__ == '__main__':
    eval()