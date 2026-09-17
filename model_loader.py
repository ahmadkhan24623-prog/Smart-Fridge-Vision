import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import numpy as np

# 22 Core Pantry & Kitchen Ingredients
CLASSES = [
    "chicken", "beef", "fish", "rice", "egg", "tomato", "onion", "garlic", 
    "potato", "bread", "milk", "butter", "cheese",
    "carrot", "bell pepper", "pasta", "lemon",
    "spinach", "mushroom", "yogurt", "oil", "flour"
]

class IngredientDetector:
    def __init__(self, model_weights_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = CLASSES
        
        # Pre-trained ImageNet Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=weights)
        self.categories = weights.meta["categories"]
        
        self.model.to(self.device)
        self.model.eval()

        self.transform = weights.transforms()

    def predict(self, pil_image, top_k=5):
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)

        # Top 15 predicted ImageNet classes
        top_p, top_i = torch.topk(probs, 15)
        top_dict = {self.categories[top_i[j].item()].lower(): top_p[j].item() for j in range(15)}
        top_labels = list(top_dict.keys())
        top_raw_label = top_labels[0]

        # Visual color and appearance cues
        arr = np.array(pil_image.resize((100, 100)))
        r, g, b = arr.mean(axis=(0, 1))

        # Color analytics
        # Beef: Deep dark red / maroon (R significantly higher than G and B)
        is_deep_red = (r > 90 and (r - g) > 25 and (r - b) > 25)
        # Chicken: Pale pink / flesh / peach tones (G and B are high, R - G is moderate)
        is_pale_pink = (r > 140 and (r - g) <= 28 and (r - b) <= 32 and g > 130 and b > 130)
        # Curry: Warm orange-red spiced gravy
        is_curry_tone = (r > 150 and g < 130 and b < 80 and (r - g) > 40)
        # Egg yolk: Bright golden yellow
        is_yolk_yellow = (r > 170 and g > 130 and b < 100 and (r - b) > 60)

        # -------------------------------------------------------------
        # High-Precision Culinary Visual Detection
        # -------------------------------------------------------------

        # 1. Fish & Seafood (Pan-Fried Fish Steaks, Cutlets, Fillets, Skillet Seafood)
        seafood_terms = [
            'fish', 'salmon', 'coho', 'sturgeon', 'gar', 'barracouta', 'eel', 'tench', 
            'goldfish', 'puffer', 'ray', 'stingray', 'crab', 'lobster', 'conch', 'crayfish',
            'anemone fish', 'lionfish'
        ]
        has_seafood_term = any(any(st in lbl for st in seafood_terms) for lbl in top_labels[:15])
        has_pan = any(p in top_dict and top_dict[p] > 0.04 for p in ['frying pan', 'wok'])
        is_fish_scene = (has_seafood_term and (has_pan or 'plate' in top_labels[:3] or any(f in top_labels[:5] for f in ['fish', 'salmon', 'tench', 'gar', 'sturgeon'])))

        if is_fish_scene or ('frying pan' in top_dict and top_dict['frying pan'] > 0.10 and has_seafood_term):
            pantry = ['fish', 'lemon', 'garlic', 'oil', 'flour']
            return pantry[:top_k], 'Crispy Pan-Seared Fish / Seafood', False

        # 2. Beef & Red Meat (Raw steaks on plate/tray, beef cutlets, meat loaf)
        beef_terms = ['meat loaf', 'cheeseburger', 'steak', 'beef', 'butcher shop', 'ox', 'bison', 'bull', 'water buffalo']
        has_beef_term = any(any(bt in lbl for bt in beef_terms) for lbl in top_labels[:8])
        is_meat_platter = ('tray' in top_labels[:5] or 'plate' in top_labels[:5] or 'butcher shop' in top_labels[:8])
        if (has_beef_term and is_deep_red) or (is_meat_platter and is_deep_red) or ('meat loaf' in top_dict and top_dict['meat loaf'] > 0.08):
            pantry = ['beef', 'garlic', 'potato', 'butter', 'oil']
            return pantry[:top_k], 'Fresh Beef Steak / Red Meat', False

        # 3. Rice & Grain Dishes (Fried rice, pulao, steamed rice, wok grain dishes)
        rice_kws = ['rice', 'grain', 'burrito', 'carbonara', 'wok', 'corn']
        has_rice_term = any(any(rw in lbl for rw in rice_kws) for lbl in top_labels[:6])
        is_wok_or_bowl = any(w in top_dict and top_dict[w] > 0.08 for w in ['wok', 'mixing bowl', 'soup bowl'])
        is_rice_dish = has_rice_term or is_wok_or_bowl or ('wok' in top_labels[:3]) or ('carbonara' in top_labels[:4])
        if is_rice_dish:
            pantry = ['rice', 'garlic', 'onion', 'oil', 'carrot', 'egg']
            return pantry[:top_k], 'Fragrant Fried Rice / Rice Dish', False

        # 4. Chicken & Poultry Detection (Raw pieces in tray, cooked curry, rotisserie)
        poultry_kws = ['chicken', 'rooster', 'poultry', 'partridge', 'dorking', 'quail']
        poultry_excl = ['cockroach', 'cocktail', 'cockatoo', 'cocker', 'hen-of-the-woods']
        has_poultry_term = any(any(kw in lbl for kw in poultry_kws) and not any(ex in lbl for ex in poultry_excl) for lbl in top_labels[:10])
        # Hen keyword check specifically excluding hen-of-the-woods mushroom
        has_real_hen = any('hen' in lbl and 'woods' not in lbl for lbl in top_labels[:5])
        has_cooked_chicken = ('dutch oven' in top_dict and top_dict['dutch oven'] > 0.05) or (is_curry_tone and any(top_dict.get(w, 0) > 0.04 for w in ['potpie', 'hot pot', 'dutch oven']))
        tray_or_pack = any(w in top_labels[:5] for w in ['tray', 'plate', 'platter', 'tub'])

        if has_cooked_chicken:
            pantry = ['chicken', 'onion', 'tomato', 'garlic', 'oil']
            return pantry[:top_k], 'Chicken Curry / Karahi', False

        # Real raw chicken requires pinkish flesh tone with distinct red dominance, not neutral white/cream
        is_real_raw_chicken = is_pale_pink and (r - g >= 16) and (r - b >= 14)
        if has_poultry_term or has_real_hen or (tray_or_pack and is_real_raw_chicken):
            pantry = ['chicken', 'garlic', 'onion', 'oil', 'tomato']
            return pantry[:top_k], 'Fresh Chicken Cuts / Poultry', False

        # 5. Egg Dishes & Breakfast (Fried eggs, boiled eggs, omelet, scrambled)
        egg_kws = ['eggnog', 'custard', 'scrambled', 'omelet']
        has_egg_term = any(w in top_dict and top_dict[w] > 0.03 for w in egg_kws)
        has_fried_egg_visual = (is_yolk_yellow and any(w in top_labels[:5] for w in ['spotlight', 'plate', 'frying pan', 'ping-pong ball']))
        has_carton = any(w in top_labels[:4] for w in ['carton', 'crate'])
        if has_egg_term or has_fried_egg_visual or has_carton:
            pantry = ['egg', 'butter', 'milk', 'tomato', 'cheese']
            return pantry[:top_k], 'Fresh Farm Eggs / Breakfast Dish', False

        # 6. Potato Dishes (Mashed Potatoes, Fries, Baked Potato)
        if any(w in top_labels[:3] for w in ['mashed potato', 'french fries', 'potato']) and top_p[0] > 0.15:
            pantry = ['potato', 'onion', 'oil', 'salt']
            return pantry[:top_k], 'Crispy Potato Dish', False

        # 7. Salads & Raw Vegetables (Only if high confidence in top 3, not raw meat)
        veg_kws = ['bell pepper', 'cucumber', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini', 'artichoke', 'mushroom', 'spinach', 'salad']
        has_veg = any(any(v in lbl for v in veg_kws) and top_dict[lbl] > 0.15 for lbl in top_labels[:3])
        if has_veg and not is_deep_red and not is_pale_pink:
            pantry = ['tomato', 'cheese', 'oil', 'onion', 'bell pepper']
            return pantry[:top_k], 'Fresh Vegetable Salad', False

        # 8. Bread & Bakery
        if any(w in top_labels[:3] for w in ['french loaf', 'bagel', 'pretzel', 'bakery']) and top_p[0] > 0.15:
            pantry = ['bread', 'butter', 'cheese', 'egg']
            return pantry[:top_k], 'Bread & Bakery', False

        # -------------------------------------------------------------
        # Unrecognized / Unsupported Non-Fridge Item
        # When uploaded image is not a recognized fridge pantry food item
        # -------------------------------------------------------------
        return [], top_raw_label, True