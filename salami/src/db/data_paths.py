from typing import List 

class DataPaths: 
    def __init__(self):
        self.base_path = r"salami/data"
        
    def get_data_path_by_day(self, day: int) -> List[str]: 
        self.validate_day(day) 
        
        if (day < 10): 
            day = f"0{day}" 

        annotation_data = self.build_annotation_data_path(day) 
        color_data = self.build_color_data_path(day) 
        multispectral_data = self.build_multispectral_path(day) 
        return [annotation_data, color_data, multispectral_data] 
    
    def validate_day(self, day): 
        if not ( (day == 1) or (day == 6) or (day == 13) or (day == 28)): 
            raise AttributeError("Day does not match any existing days. Choose either 1,6,13,20 or 28") 
        
    def build_annotation_data_path(self, day): 
        return f"{self.base_path}/annotation_day{day}.png" 
    
    def build_color_data_path(self, day): 
        return f"{self.base_path}/color_day{day}.png" 
    
    def build_multispectral_path(self, day):
        return f"{self.base_path}/multispectral_day{day}.mat"
    