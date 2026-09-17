import pandas as pd

PRIMARY_PROTEINS = {"chicken", "beef", "fish", "egg"}

SYNONYMS = {
    "olive-oil": "oil",
    "olive oil": "oil",
    "extra virgin olive oil": "oil",
    "potatoes": "potato",
    "tomatoes": "tomato",
    "onions": "onion",
    "eggs": "egg"
}

def normalize_ingredient(ing):
    cleaned = ing.strip().lower()
    return SYNONYMS.get(cleaned, cleaned)

import os

class RecipeRecommender:
    def __init__(self, dataset_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_path = None
        
        # 1. If explicit dataset_path provided and exists
        if dataset_path:
            if os.path.exists(dataset_path):
                resolved_path = dataset_path
            elif os.path.exists(os.path.join(base_dir, dataset_path)):
                resolved_path = os.path.join(base_dir, dataset_path)
                
        # 2. Check standard candidate names
        if not resolved_path:
            for fname in ["recipes_backup.csv", "recipes.csv"]:
                p = os.path.join(base_dir, fname)
                if os.path.exists(p):
                    resolved_path = p
                    break
                if os.path.exists(fname):
                    resolved_path = fname
                    break

        # 3. Dynamic scan: Find ANY .csv file in project directory or current directory
        if not resolved_path:
            for folder in [base_dir, "."]:
                if os.path.exists(folder):
                    for f in os.listdir(folder):
                        if f.lower().endswith(".csv") and not f.startswith("."):
                            resolved_path = os.path.join(folder, f)
                            break
                    if resolved_path:
                        break

        if not resolved_path or not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Could not locate recipe dataset (.csv). Checked {base_dir} and current working directory."
            )

        self.dataset_path = resolved_path
        self.df = pd.read_csv(resolved_path)
        # Ingredients string ko Python set me convert karein
        self.df["ing_set"] = self.df["ingredients"].apply(
            lambda x: set([normalize_ingredient(i) for i in str(x).split(",") if i.strip()])
        )

    def recommend(self, user_ingredients, top_k=25, min_match=30):
        if not user_ingredients:
            return pd.DataFrame()

        user_set = set([normalize_ingredient(i) for i in user_ingredients])
        user_proteins = user_set.intersection(PRIMARY_PROTEINS)
        has_rice_focus = ("rice" in user_set and not user_proteins)
        results = []

        for _, row in self.df.iterrows():
            recipe_set = row["ing_set"]
            common = user_set.intersection(recipe_set)
            missing = recipe_set - user_set

            # Primary Protein Integrity:
            # Agar recipe me chicken/beef/fish/egg mandatory hai, aur user ke paas woh protein nahi hai,
            # toh recipe recommend mat karein
            recipe_proteins = recipe_set.intersection(PRIMARY_PROTEINS)
            if recipe_proteins:
                if not user_proteins.intersection(recipe_proteins):
                    continue
            elif user_proteins:
                # If user has a dedicated protein selected, skip non-protein recipes unless high match
                pass

            # Check if recipe has user's selected primary protein or rice
            has_user_protein = bool(user_proteins and user_proteins.intersection(recipe_proteins))
            has_user_rice = bool(has_rice_focus and "rice" in recipe_set)

            if has_user_protein:
                # Protein recipes: Having the main protein is 50% base score + up to 50% for complementary ingredients
                staples_in_rec = recipe_set - PRIMARY_PROTEINS
                common_staples = common - PRIMARY_PROTEINS
                staple_ratio = len(common_staples) / len(staples_in_rec) if staples_in_rec else 1.0
                match_score = round((0.50 + 0.50 * staple_ratio) * 100, 1)
            elif has_user_rice:
                staples_in_rec = recipe_set - {"rice"}
                common_staples = common - {"rice"}
                staple_ratio = len(common_staples) / len(staples_in_rec) if staples_in_rec else 1.0
                match_score = round((0.50 + 0.50 * staple_ratio) * 100, 1)
            else:
                # Standard Jaccard proportion for other recipes
                match_score = round((len(common) / len(recipe_set)) * 100, 1) if len(recipe_set) > 0 else 0

            # Qualified recipes
            if has_user_protein or has_user_rice or match_score >= min_match:
                results.append({
                    "title": row.get("title", "Recipe"),
                    "category": row.get("category", "General"),
                    "prep_time": row.get("prep_time", "15 mins"),
                    "cook_time": row.get("cook_time", "20 mins"),
                    "servings": row.get("servings", "2-3 servings"),
                    "difficulty": row.get("difficulty", "Easy"),
                    "match_score": match_score,
                    "has_protein": has_user_protein or has_user_rice,
                    "matched_ingredients": sorted(list(common)),
                    "missing_ingredients": sorted(list(missing)),
                    "quantities": row.get("quantities", ""),
                    "instructions": row.get("instructions", ""),
                    "chef_tip": row.get("chef_tip", "")
                })

        if not results:
            return pd.DataFrame()

        res_df = pd.DataFrame(results)
        # Prioritize recipes containing user's primary protein, then sort by highest match score
        res_df = res_df.sort_values(by=["has_protein", "match_score"], ascending=[False, False])
        return res_df.head(top_k)