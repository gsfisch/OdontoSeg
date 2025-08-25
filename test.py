import os
import torch
from util.data import get_test_generator
from util.model import make_model
from loops import test_loop
from optimizers.main import optimizer
from config import training_config, path_models
from util.scheduler import FlatplusAnneal
import wandb

def load_model(model_path):
    """
    Carrega o modelo a partir do caminho fornecido.
    """
    print(f'Carregando modelo: {model_path}')
    model = make_model(training_config['encoder'], training_config['architecture'], classes=training_config['classes'])
    model.load_state_dict(torch.load(model_path))  # Carrega os pesos do modelo
    model.cuda()
    model.eval()
    print(f'Carregadooo')
    return model

def evaluate_model(model, test_generator):
    """
    Avalia o modelo usando o gerador de dados de teste e retorna a perda, acurácia e mIoU.
    """
    with torch.no_grad():
        metrics = test_loop(test_generator, model)
        return metrics

def save_best_models(best_metrics, folder_experiment):
    """
    Salva os melhores modelos para cada métrica em arquivos distintos.
    """
    for metric, (best_value, best_model_path, index) in best_metrics.items():
        if best_model_path:
            best_model_save_path = os.path.join(folder_experiment, f"best_model_by_{metric}_epoch_{index}.pth")
            model = load_model(best_model_path)
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Saved best model for {metric} to {best_model_save_path}")

def test_routine(folder_experiment:str, best_epoch_training: int, wand_logged:bool = False):
    print('\nINITIALIZATING TEST ROUTINE\n')
    print(f'folder_experiment: {folder_experiment}')
    torch.cuda.empty_cache()

    best_metrics = {
        "loss": (float('inf'), None, None),
        "accuracy": (-float('inf'), None, None),
        "mIoU": (-float('inf'), None, None),
        "precision": (-float('inf'), None, None),
        "recall": (-float('inf'), None, None),
        "dice": (-float('inf'), None, None),
    }
    
    test_generator = get_test_generator()

    experiments = [os.path.join(folder_experiment, f) for f in os.listdir(folder_experiment) if os.path.isfile(os.path.join(folder_experiment, f))]
    metrics = []
    for index, experiment in enumerate(experiments):
        if not os.path.isfile(experiment):
            print(f"File not found: {experiment}")
            continue
        
        try:
            model = load_model(experiment)
            current_metrics = evaluate_model(model, test_generator)
            if wand_logged:
                print('LOGANDO')
                # Log metrics to WandB
                wandb.log({
                    "test/loss": current_metrics['loss'],
                    "test/accuracy": current_metrics['accuracy'],
                    "test/mIoU": current_metrics['mIoU'],
                    "test/precision": current_metrics['precision'],
                    "test/recall":current_metrics['recall'],
                    "test/dice": current_metrics['dice'],
                })

            # Update best metrics
            for key in best_metrics:
                if key in current_metrics:
                    if (key in ["loss"] and current_metrics[key] < best_metrics[key][0]) or \
                       (key not in ["loss"] and current_metrics[key] > best_metrics[key][0]):
                        best_metrics[key] = (current_metrics[key], experiment, index)
                        
            # Store metrics for later evaluation
            metrics.append((experiment, current_metrics))
            print(f"\nIndex: {index}, Test File: {experiment}")
            print(f"Metrics: {current_metrics}\n")
        except Exception as e:
            print(f"Error processing {experiment}: {e}")
    
    save_best_models(best_metrics, folder_experiment)
    
    # Delete files that are not the best models
    for experiment in experiments:
        if best_epoch_training == experiment:
            continue  # Do not delete the best epoch's model
        os.remove(experiment)
    
if __name__ == "__main__":
    folder_experiment = "unet_vgg19_focal_loss_20240905-154617"
    experiment_path = os.path.join(path_models, folder_experiment)
    test_routine(experiment_path, 499)