import os

DEVICE = None
SPECIAL_TOKENS = {
    '<user>',
    '<number>',
    '<url>',
    '<emoji>',
    '<punctuation>',
}

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__))))))

EXP_DIR  = os.path.join(PROJ_DIR, 'exp_ssl')

# OUT-OF-SOURCE DIRECTORIES
CHECKPOINT_DIR  = os.path.join(EXP_DIR, 'checkpoints')
REPORT_DIR      = os.path.join(EXP_DIR, 'reports')
DATA_DIR        = os.path.join(PROJ_DIR, 'data')

# SUBFOLDERS OF REPORT
FIGURE_DIR      = os.path.join(REPORT_DIR, 'figures')
HISTORY_DIR     = os.path.join(REPORT_DIR, 'history')
PREDICTIONS_DIR = os.path.join(REPORT_DIR, 'predictions')

# EVALUATION DIRECTORY FOR NON-PYTHON SCRIPTS
EVAL_DIR        = os.path.join(EXP_DIR, 'src/evaluation')

