from .context_data import context_data_load, context_data_split, context_data_loader
from .dl_data import dl_data_load, dl_data_split, dl_data_loader
from .image_data import image_data_load, image_data_split, image_data_loader
from .text_data import text_data_load, text_data_split, text_data_loader
#이게 있어서 이 디렉을 가져오는 것만으로도 이 함수들을 사용할 수 있게 된다.

from .donggun_data import donggun_data_load, donggun_data_loader, donggun_data_split