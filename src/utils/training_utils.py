import os

from utils import LOGGER, TQDM



def init_progress_bar(dloader, is_rank_zero, loss_names, nb):
    if is_rank_zero:
        header = tuple(['Epoch'] + loss_names)
        LOGGER.info(('\n' + '%15s' * (1 + len(loss_names))) % header)
        pbar = TQDM(enumerate(dloader), total=nb)
    else:
        pbar = enumerate(dloader)
    return pbar


def choose_proper_resume_model(resume_dir, type):
    weights_dir = os.listdir(os.path.join(resume_dir, 'weights'))
    try:
        weight = list(filter(lambda x: type in x, weights_dir))[0]
        return os.path.join(resume_dir, 'weights', weight)
    except IndexError:
        raise IndexError(f"There's no model path in {weights_dir} of type {type}")