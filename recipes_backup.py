"""
Backup Recipe Dataset and Fallback Recommender Module.
Provides guaranteed in-memory recipe data if external files or network services are unavailable.
"""

import os
import pandas as pd

# Standard 37 Curated Backup Recipes
BACKUP_RECIPES = [
    {
        "title": "Classic Garlic Butter Beef Steak",
        "category": "Non-Veg",
        "prep_time": "10 mins",
        "cook_time": "15 mins",
        "servings": "2 servings",
        "difficulty": "Medium",
        "ingredients": "beef,butter,garlic,potato,oil",
        "quantities": "500g Beef Steak (Ribeye or Sirloin cut)|3 tbsp Butter (Unsalted)|4 cloves Garlic (smashed)|2 medium Potatoes (baked or mashed)|2 tbsp Olive Oil or Cooking Oil|1 tsp Salt & 1 tsp Coarse Black Pepper|Fresh rosemary or thyme sprigs (optional garnish)",
        "instructions": "Step 1: Bring beef steak to room temperature 20 minutes before cooking. Pat dry thoroughly with paper towels.|Step 2: Season both sides generously with salt and freshly cracked black pepper.|Step 3: Heat 2 tbsp oil in a heavy cast-iron skillet over high flame until smoking hot.|Step 4: Carefully lay down the steak. Sear untouched for 3 to 4 minutes until a deep, caramelized golden-brown crust forms before flipping.|Step 5: Flip the steak. Immediately drop in 3 tbsp butter, smashed garlic cloves, and fresh rosemary sprigs into the pan.|Step 6: Tilt the pan slightly and continuously spoon the sizzling foaming garlic-herb butter over the steak for 2-3 minutes.|Step 7: Transfer steak to a warm cutting board and let rest for 5-8 minutes to lock in the juices. Slice against the grain and serve alongside warm baked potatoes.",
        "chef_tip": "Resting the steak for 5-8 minutes after cooking is non-negotiable! It redistributes the juices so every bite is tender and succulent."
    },
    {
        "title": "Crispy Garlic Herb Roast Potatoes",
        "category": "Vegetarian",
        "prep_time": "10 mins",
        "cook_time": "30 mins",
        "servings": "3-4 servings",
        "difficulty": "Easy",
        "ingredients": "potato,garlic,butter,oil",
        "quantities": "750g Russet Potatoes (cubed)|3 tbsp Butter|4 cloves Garlic (minced)|2 tbsp Olive Oil|1 tsp Salt|1 tsp Dried Italian Herbs & Black Pepper",
        "instructions": "Step 1: Preheat oven or skillet. Parboil cubed potatoes in salted water for 5 minutes, then drain and shake vigorously in the colander to roughen edges.|Step 2: Heat olive oil and 1 tbsp butter in a pan over medium heat.|Step 3: Toss potatoes in the pan, spreading in a single layer. Fry or roast for 20 minutes until golden and crispy.|Step 4: In the last 3 minutes, add minced garlic and remaining butter, tossing until aromatic and glazed.|Step 5: Season with sea salt, black pepper, and herbs. Serve piping hot with garlic butter drizzle.",
        "chef_tip": "Roughening the potato edges after parboiling creates maximum surface area for that ultra-crispy restaurant crunch."
    },
    {
        "title": "Pan-Seared Buttered Steak Bites",
        "category": "Non-Veg",
        "prep_time": "5 mins",
        "cook_time": "10 mins",
        "servings": "2-3 servings",
        "difficulty": "Easy",
        "ingredients": "beef,garlic,butter,oil",
        "quantities": "500g Sirloin or Ribeye Beef (cut into 1-inch cubes)|3 tbsp Butter|4 cloves Garlic (minced)|2 tbsp Cooking Oil|1/2 tsp Salt & 1/2 tsp Coarse Black Pepper|1 tbsp chopped fresh parsley",
        "instructions": "Step 1: Pat beef cubes dry with paper towels and season with salt and coarse black pepper.|Step 2: Heat 2 tbsp oil in a large cast-iron skillet over smoking high heat.|Step 3: Add beef cubes in a single layer (do not overcrowd) and sear undisturbed for 2 minutes to form a deep crust.|Step 4: Stir and cook for another 2 minutes until browned on all sides.|Step 5: Reduce heat to medium, add butter and minced garlic. Baste the steak bites in the foaming garlic butter for 1 minute.|Step 6: Remove from heat immediately, garnish with fresh parsley, and serve with toothpicks or over rice.",
        "chef_tip": "High heat and quick cooking are the secrets to juicy steak bites that stay tender inside."
    },
    {
        "title": "Chicken Curry",
        "category": "Non-Veg",
        "prep_time": "15 mins",
        "cook_time": "30 mins",
        "servings": "4 servings",
        "difficulty": "Medium",
        "ingredients": "chicken,onion,tomato,garlic,oil",
        "quantities": "500g Chicken (bone-in or boneless cut into pieces)|2 medium Onions (finely chopped)|3 ripe Tomatoes (pureed or finely diced)|1 tbsp Garlic paste (4-5 cloves)|3 tbsp Cooking Oil|1 tsp Salt|1 tsp Turmeric powder|1 tsp Red Chili powder|1 tsp Garam Masala|Fresh Coriander for garnish",
        "instructions": "Step 1: Heat 3 tbsp oil in a deep pan or pot over medium flame. Add finely chopped onions and sauté for 6-8 minutes until golden brown.|Step 2: Add 1 tbsp minced garlic (and ginger if available) and sauté for 1 minute until fragrant.|Step 3: Add chicken pieces and sear on high heat for 5 minutes until the chicken changes color to white and is lightly browned.|Step 4: Pour in the pureed tomatoes along with salt, turmeric, red chili, and cumin powder. Stir well and cook until oil starts separating from the gravy (about 6 minutes).|Step 5: Add 1 cup of warm water, cover with a lid, and simmer on low-medium heat for 15-18 minutes until the chicken is tender and cooked through.|Step 6: Sprinkle garam masala and chopped fresh coriander on top. Turn off heat, let it rest for 2 minutes, and serve hot with naan, roti, or steamed basmati rice.",
        "chef_tip": "For restaurant-style rich gravy, brown the onions thoroughly before adding garlic and chicken. A dollop of yogurt whisked in at Step 4 adds extra creaminess!"
    },
    {
        "title": "Chicken Karahi",
        "category": "Non-Veg",
        "prep_time": "10 mins",
        "cook_time": "25 mins",
        "servings": "3-4 servings",
        "difficulty": "Easy",
        "ingredients": "chicken,tomato,garlic,oil,salt",
        "quantities": "600g Chicken cut into medium pieces|4 large ripe Tomatoes (halved)|1 tbsp freshly minced Garlic|4 tbsp Cooking Oil or Ghee|1 tsp Salt|1 tbsp Crushed Black Pepper|1 tbsp Coriander seeds (crushed)|2-3 fresh Green Chilies (slit)|Fresh ginger juliennes for garnish",
        "instructions": "Step 1: Heat 4 tbsp oil in a wok (karahi) on high heat. Add the chicken pieces and 1 tsp salt. Fry for 5-7 minutes until chicken turns light golden.|Step 2: Add 1 tbsp minced garlic and stir-fry for 1-2 minutes until aromatic.|Step 3: Place tomato halves skin-side up over the chicken. Cover with a lid and let steam on medium flame for 5 minutes until tomato skins soften.|Step 4: Remove lid, peel off tomato skins easily with tongs, and mash the tomato pulp into the chicken using a spatula.|Step 5: Turn flame to high. Continuously stir-fry until tomato liquid evaporates and oil clearly separates from the thick tomato masala.|Step 6: Stir in crushed black pepper, green chilies, and crushed coriander seeds. Cook for 2 final minutes.|Step 7: Garnish generously with fresh ginger juliennes and cilantro. Serve piping hot directly in the karahi with tandoori naan.",
        "chef_tip": "Never add water to a traditional Karahi! The chicken cooks entirely in the natural juices of the tomatoes and high-heat stir frying."
    },
    {
        "title": "Creamy Garlic Chicken",
        "category": "Non-Veg",
        "prep_time": "10 mins",
        "cook_time": "20 mins",
        "servings": "2-3 servings",
        "difficulty": "Easy",
        "ingredients": "chicken,garlic,butter,milk,cheese",
        "quantities": "450g Boneless Chicken breast (cut into bite-sized cutlets)|2 tbsp Butter|5 cloves Garlic (minced)|3/4 cup Milk (or heavy cream)|1/2 cup Shredded Cheese (cheddar or mozzarella)|1 tbsp Cooking Oil|1/2 tsp Salt & 1/2 tsp Black Pepper|1 tsp dried Oregano or Italian herbs",
        "instructions": "Step 1: Season chicken pieces on both sides with salt, black pepper, and oregano.|Step 2: Heat 1 tbsp oil and 1 tbsp butter in a skillet over medium-high heat. Add chicken and sear for 4-5 minutes per side until golden and cooked through. Transfer to a plate.|Step 3: In the same skillet, melt remaining 1 tbsp butter over medium heat. Add minced garlic and sauté for 1 minute until fragrant (do not burn).|Step 4: Lower the flame and gradually pour in milk, whisking constantly to deglaze the flavorful pan drippings.|Step 5: Stir in the shredded cheese and stir until melted into a velvety, smooth sauce.|Step 6: Return the cooked chicken to the skillet, spooning the rich garlic cheese sauce over the top. Simmer gently for 2 minutes.|Step 7: Serve warm over pasta, mashed potatoes, or with crusty garlic bread.",
        "chef_tip": "Use whole milk or full cream for the smoothest sauce. Add a pinch of parmesan if you like an extra savory kick."
    },
    {
        "title": "Crispy Fried Fish",
        "category": "Seafood",
        "prep_time": "15 mins",
        "cook_time": "12 mins",
        "servings": "3-4 servings",
        "difficulty": "Easy",
        "ingredients": "fish,lemon,garlic,oil,flour",
        "quantities": "750g Fish fillets or steaks (Rohu, Surmai, or Tilapia)|2 tbsp Lemon juice|1 tbsp Garlic paste|3 tbsp Gram flour (besan) or corn flour|3 tbsp Cooking Oil for frying|1 tsp Carom seeds (ajwain)|1 tsp Red chili powder & 1 tsp Cumin|1 tsp Chaat Masala",
        "instructions": "Step 1: Pat fish cutlets or steaks completely dry with paper towels to ensure a crispy coating.|Step 2: In a bowl, mix lemon juice, garlic paste, crushed carom seeds (ajwain), red chili powder, cumin, salt, and gram flour with a splash of water into a thick spice batter.|Step 3: Coat each fish cutlet thoroughly with the marinade and let rest for 15 minutes.|Step 4: Heat cooking oil in a wide frying pan over medium-high heat.|Step 5: Place fish pieces in the hot oil without overcrowding. Fry for 4-5 minutes on the first side until golden and crunchy.|Step 6: Carefully flip and fry the other side for 3-4 minutes until cooked through and flakey.|Step 7: Drain on paper towels, sprinkle with tangy chaat masala and fresh lemon juice, and serve hot!",
        "chef_tip": "Crushed ajwain (carom seeds) and lemon juice cut any fishy odor and give that signature authentic restaurant aroma."
    },
    {
        "title": "Traditional Fish Curry",
        "category": "Seafood",
        "prep_time": "15 mins",
        "cook_time": "20 mins",
        "servings": "4 servings",
        "difficulty": "Medium",
        "ingredients": "fish,onion,tomato,garlic,oil",
        "quantities": "600g Fish cutlets or steaks|2 medium Onions (finely grated)|2 ripe Tomatoes (pureed)|1 tbsp Garlic paste|3 tbsp Cooking Oil|1/2 tsp Fenugreek seeds (methi dana)|1 tsp Turmeric & 1 tsp Coriander powder|Fresh coriander and green chilies for garnish",
        "instructions": "Step 1: Lightly sear fish cutlets in 1 tbsp oil for 2 minutes per side in a wide pan, then set aside.|Step 2: In the same pan, heat remaining oil, crackle fenugreek seeds, and sauté grated onions until golden brown.|Step 3: Add garlic paste, turmeric, coriander, and pureed tomatoes. Cook on medium heat until oil separates from the masala.|Step 4: Pour in 1.5 cups of warm water and bring the curry to a gentle simmer.|Step 5: Carefully slide the fish cutlets into the simmering sauce. Cover and cook on low heat for 8-10 minutes.|Step 6: Gently swirl the pan by the handles (do not stir aggressively with a spoon) to prevent breaking the fish.|Step 7: Garnish with cilantro and green chilies. Serve hot with steamed rice.",
        "chef_tip": "Never vigorously stir fish curry with a spoon while cooking! Delicate fish breaks easily, so gently swirl the pan instead."
    },
    {
        "title": "Lemon Garlic Butter Fish",
        "category": "Seafood",
        "prep_time": "10 mins",
        "cook_time": "12 mins",
        "servings": "2-3 servings",
        "difficulty": "Easy",
        "ingredients": "fish,butter,garlic,lemon",
        "quantities": "500g Fish steaks or fillets|3 tbsp Butter|4 cloves Garlic (finely minced)|2 tbsp Fresh Lemon juice|1/4 tsp Salt & 1/4 tsp Cracked Black Pepper|1 tbsp fresh chopped parsley",
        "instructions": "Step 1: Season fish steaks generously with salt and black pepper on both sides.|Step 2: Melt 1.5 tbsp butter in a non-stick skillet over medium-high heat.|Step 3: Add the fish steaks and sear for 3-4 minutes until golden on the bottom.|Step 4: Flip carefully and cook the other side for 2-3 minutes until the fish flakes easily with a fork.|Step 5: Add minced garlic and remaining butter to the pan, swirling for 1 minute until aromatic.|Step 6: Remove pan from heat, squeeze fresh lemon juice over the fish, and spoon the sizzling lemon-garlic butter sauce on top.|Step 7: Garnish with fresh parsley and serve immediately.",
        "chef_tip": "Cold butter swirled into the pan at the very end emulsifies with lemon juice for a velvety gourmet pan sauce."
    },
    {
        "title": "Traditional Beef Curry",
        "category": "Non-Veg",
        "prep_time": "20 mins",
        "cook_time": "45 mins",
        "servings": "4 servings",
        "difficulty": "Medium",
        "ingredients": "beef,onion,tomato,garlic,oil",
        "quantities": "600g Beef cut into bite-sized cubes|2 large Onions (sliced)|3 ripe Tomatoes (chopped)|1.5 tbsp Garlic paste|3 tbsp Cooking Oil|1 tsp Salt|1 tsp Turmeric & 1.5 tsp Red Chili powder|1 tsp Garam Masala|Fresh coriander and ginger juliennes",
        "instructions": "Step 1: Heat 3 tbsp oil in a pressure cooker or heavy-bottomed pot. Add sliced onions and fry for 8-10 minutes until deep golden brown.|Step 2: Add garlic paste and beef cubes. Sauté on high flame for 5-6 minutes until beef is seared on all sides.|Step 3: Add chopped tomatoes, salt, turmeric, and chili powder. Cook until tomatoes are soft and oil separates.|Step 4: Pour in 2 cups of water. Cover and pressure cook for 25-30 minutes (or simmer covered on low heat for 50 minutes) until beef is melt-in-the-mouth tender.|Step 5: Open lid, simmer on medium heat to thicken gravy to desired consistency.|Step 6: Sprinkle garam masala, fresh coriander, and ginger juliennes on top. Serve hot with warm naan or roti.",
        "chef_tip": "Cooking beef low and slow breaks down the connective tissues, making the gravy rich and the meat tender."
    },
    {
        "title": "Beef Pepper Stir Fry",
        "category": "Non-Veg",
        "prep_time": "15 mins",
        "cook_time": "15 mins",
        "servings": "3 servings",
        "difficulty": "Easy",
        "ingredients": "beef,bell pepper,onion,garlic,oil,rice",
        "quantities": "400g Tender Beef steak (thinly sliced across grain)|1 large Bell Pepper (sliced into strips)|1 medium Onion (cut into wedges)|3 cloves Garlic (minced)|2 tbsp Cooking Oil|2 tbsp Soy Sauce|1/2 tsp Black Pepper|2 cups Steamed Rice for serving",
        "instructions": "Step 1: Slice beef very thinly against the grain. Toss with 1 tbsp soy sauce and black pepper.|Step 2: Heat 1 tbsp oil in a wide wok over high heat. Add beef slices in a single layer and sear for 2 minutes without stirring until caramelized. Stir and cook 1 more minute, then transfer to a plate.|Step 3: Add remaining 1 tbsp oil to the hot wok. Add minced garlic, sliced onions, and bell peppers. Stir-fry vigorously on high flame for 2-3 minutes until crisp-tender.|Step 4: Return the seared beef and its juices to the wok. Drizzle remaining soy sauce.|Step 5: Toss everything together on high heat for 1 minute until glossy and combined.|Step 6: Serve immediately over warm steamed rice.",
        "chef_tip": "Slicing beef thinly against the grain guarantees tender stir-fried beef that cooks in just 3 minutes."
    },
    {
        "title": "Chicken Fried Rice",
        "category": "Non-Veg",
        "prep_time": "15 mins",
        "cook_time": "15 mins",
        "servings": "3 servings",
        "difficulty": "Easy",
        "ingredients": "chicken,rice,carrot,garlic,oil,egg",
        "quantities": "2 cups Cooked Rice (cold or day-old preferred)|200g Chicken breast (finely diced)|2 Eggs (lightly beaten)|1 medium Carrot (finely diced)|3 cloves Garlic (minced)|2 tbsp Cooking Oil|1 tbsp Soy Sauce|1/2 tsp Salt & 1/2 tsp White Pepper|Chopped green onions",
        "instructions": "Step 1: Heat 1 tbsp oil in a wide wok or large pan over high heat. Add beaten eggs and scramble quickly for 30 seconds until just set. Remove and set aside.|Step 2: Add remaining 1 tbsp oil to the wok. Add diced chicken and stir-fry on high flame for 3-4 minutes until cooked through and lightly browned.|Step 3: Toss in minced garlic and diced carrots. Stir-fry vigorously for 1-2 minutes until crisp-tender.|Step 4: Add the cold cooked rice, breaking up any clumps with your spatula. Stir-fry on high heat for 2 minutes.|Step 5: Drizzle soy sauce evenly around the rim of the wok and season with salt and white pepper. Toss everything together for 1-2 minutes until thoroughly heated.|Step 6: Fold back in the scrambled eggs and chopped green onions. Toss for 30 seconds and serve hot!",
        "chef_tip": "Using day-old refrigerated rice is the golden rule! Fresh hot rice releases moisture and turns mushy when stir-fried."
    },
    {
        "title": "Classic Scrambled Eggs",
        "category": "Breakfast",
        "prep_time": "5 mins",
        "cook_time": "5 mins",
        "servings": "2 servings",
        "difficulty": "Easy",
        "ingredients": "egg,butter,salt,milk",
        "quantities": "4 large Fresh Eggs|2 tbsp Milk|1.5 tbsp Butter|1/4 tsp Salt|Pinch of Freshly Ground Black Pepper|Chopped chives or parsley (optional)",
        "instructions": "Step 1: Crack 4 eggs into a bowl, add 2 tbsp milk, salt, and pepper. Whisk vigorously for 30 seconds until completely uniform and slightly frothy.|Step 2: Melt 1.5 tbsp butter in a non-stick skillet over low-medium heat until foaming but not browned.|Step 3: Pour in egg mixture. Let it sit untouched for 20 seconds until the edges barely begin to set.|Step 4: Using a silicone spatula, gently sweep across the pan in long strokes to form soft, pillowy curds.|Step 5: Remove pan from heat while eggs still look slightly glossy and underdone (residual heat finishes cooking them perfectly).|Step 6: Transfer immediately to plates. Top with fresh black pepper and herbs. Serve with warm buttered toast.",
        "chef_tip": "Low and slow heat is the secret to velvety, creamy scrambled eggs that never dry out."
    },
    {
        "title": "Tomato Cheese Omelet",
        "category": "Breakfast",
        "prep_time": "5 mins",
        "cook_time": "8 mins",
        "servings": "1-2 servings",
        "difficulty": "Easy",
        "ingredients": "egg,tomato,cheese,salt,oil",
        "quantities": "3 large Eggs|1 small ripe Tomato (seeded and finely diced)|1/3 cup Shredded Cheese|1 tbsp Cooking Oil or Butter|1/4 tsp Salt & pinch of Black Pepper|Fresh herbs (cilantro or basil)",
        "instructions": "Step 1: Whisk eggs in a bowl with salt and black pepper until well combined.|Step 2: Heat 1 tbsp oil in an 8-inch non-stick skillet over medium flame.|Step 3: Pour the beaten eggs into the hot skillet, tilting the pan so eggs spread evenly across the surface.|Step 4: When eggs are halfway set (about 2 minutes), sprinkle diced tomatoes and shredded cheese across one half of the omelet.|Step 5: Carefully fold the empty half over the filled half using a spatula to form a neat semi-circle.|Step 6: Cook for another 1 minute until the cheese melts into gooey perfection. Slide onto a plate and enjoy hot!",
        "chef_tip": "Removing excess seeds and juice from the tomato prevents the omelet from becoming watery."
    },
    {
        "title": "Shakshuka",
        "category": "Breakfast",
        "prep_time": "10 mins",
        "cook_time": "20 mins",
        "servings": "2-3 servings",
        "difficulty": "Easy",
        "ingredients": "egg,tomato,onion,garlic,pepper,oil",
        "quantities": "4 large Eggs|3 large ripe Tomatoes (chopped)|1 medium Onion (chopped)|3 cloves Garlic (minced)|1 Bell Pepper (sliced)|2 tbsp Olive Oil or Cooking Oil|1 tsp Cumin powder|1 tsp Paprika or Chili powder|Salt & Black Pepper to taste|Fresh parsley or cilantro",
        "instructions": "Step 1: Heat 2 tbsp oil in a skillet over medium heat. Add chopped onions and bell pepper; sauté for 5 minutes until soft and sweet.|Step 2: Stir in minced garlic, cumin, paprika, salt, and black pepper. Cook for 1 minute until wonderfully fragrant.|Step 3: Add the chopped tomatoes. Simmer gently for 8-10 minutes, mashing slightly, until a thick, rich tomato sauce forms.|Step 4: Use a large spoon to make 4 small wells or depressions in the simmering sauce.|Step 5: Carefully crack an egg directly into each well. Season egg tops with a pinch of salt.|Step 6: Cover the skillet with a lid, reduce flame to low, and cook for 5-8 minutes until egg whites are firmly set but yolks are still delightfully runny.|Step 7: Garnish with fresh parsley. Serve hot directly from the pan with crusty warm bread for dipping into the yolks.",
        "chef_tip": "Dip warm pita or sourdough bread straight into the spiced tomato sauce and runny egg yolks!"
    },
    {
        "title": "Garlic Fried Rice",
        "category": "Fast & Easy",
        "prep_time": "5 mins",
        "cook_time": "10 mins",
        "servings": "2 servings",
        "difficulty": "Easy",
        "ingredients": "rice,garlic,oil,egg",
        "quantities": "2.5 cups Cooked Rice (cold)|6 cloves Garlic (thinly sliced or minced)|2 tbsp Cooking Oil|1 large Egg|1/2 tsp Salt & 1/2 tsp Black Pepper|1 tbsp Soy Sauce (optional)",
        "instructions": "Step 1: Heat 2 tbsp oil in a wok or skillet over medium flame. Add sliced garlic and fry for 2-3 minutes until golden brown and crispy. Spoon out half the crispy garlic for topping later.|Step 2: Push remaining garlic to the side of the pan. Crack the egg into the center and scramble for 40 seconds.|Step 3: Add the cold cooked rice to the wok. Increase heat to high.|Step 4: Stir-fry rice continuously for 3-4 minutes, pressing out clumps so every grain is coated in garlic oil.|Step 5: Season with salt, black pepper, and optional soy sauce. Toss well for 1 minute.|Step 6: Dish out into bowls and top with the reserved golden crispy garlic chips.",
        "chef_tip": "The crispy garlic chips on top give an irresistible aroma and crunch to this classic comfort dish."
    },
    {
        "title": "Tomato Mozzarella Salad",
        "category": "Fast & Easy",
        "prep_time": "5 mins",
        "cook_time": "0 mins",
        "servings": "2 servings",
        "difficulty": "Easy",
        "ingredients": "tomato,cheese,oil",
        "quantities": "3 ripe large Tomatoes (sliced into rounds)|150g Fresh Mozzarella or white cheese (sliced)|2 tbsp Olive Oil|Pinch of Salt & Coarse Black Pepper|Fresh Basil leaves (or dried oregano)",
        "instructions": "Step 1: Wash and slice tomatoes into 1/4-inch thick uniform rounds.|Step 2: Slice the cheese into matching rounds.|Step 3: Arrange alternating slices of tomato and cheese overlapping in a circular pattern on a serving platter.|Step 4: Tuck fresh basil leaves between the tomato and cheese slices.|Step 5: Drizzle high-quality olive oil generously over the entire dish.|Step 6: Season with freshly ground black pepper and a pinch of flaky sea salt. Serve immediately chilled or at room temperature.",
        "chef_tip": "Best made with ripe, sweet in-season tomatoes. A dash of balsamic glaze adds a gourmet sweet-tangy finish."
    },
    {
        "title": "Butter Chicken (Murgh Makhani)",
        "category": "Non-Veg",
        "prep_time": "15 mins",
        "cook_time": "25 mins",
        "servings": "4 servings",
        "difficulty": "Medium",
        "ingredients": "chicken,butter,tomato,onion,garlic,yogurt",
        "quantities": "600g Boneless Chicken (cut into cubes)|3 tbsp Butter (divided)|1 cup Tomato puree|1 medium Onion (pureed)|1 tbsp Garlic paste|1/2 cup Plain Yogurt|1/2 cup Cream or Milk|1 tsp Garam Masala & 1 tsp Kashmiri Chili|1 tsp Sugar & Salt",
        "instructions": "Step 1: Marinate chicken cubes in yogurt, half the garlic paste, chili powder, and salt for 15 minutes.|Step 2: Heat 1 tbsp butter in a pan over high heat. Sear chicken cubes for 5-6 minutes until golden brown with slight char marks; remove to a plate.|Step 3: In the same pan, melt 2 tbsp butter over medium flame. Add onion puree and remaining garlic; sauté for 4-5 minutes until aromatic.|Step 4: Pour in tomato puree, garam masala, salt, and a pinch of sugar. Simmer for 8 minutes until the sauce deepens in color and butter begins to surface.|Step 5: Stir in cream or milk, creating a velvety orange-red makhani sauce.|Step 6: Return seared chicken to the simmering sauce and cook gently for 5 minutes.|Step 7: Finish with a knob of fresh butter and crushed kasuri methi (dried fenugreek). Serve hot with garlic naan or basmati rice.",
        "chef_tip": "A pinch of sugar balances the acidity of the tomatoes and elevates the rich buttery aroma to restaurant standards."
    }
]

def get_backup_df():
    """Returns the backup recipe dataset as a pandas DataFrame."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "recipes_backup.csv"),
        os.path.join(base_dir, "recipes.csv"),
        "recipes_backup.csv",
        "recipes.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                df = pd.read_csv(c)
                if len(df) > 0:
                    return df
            except Exception:
                pass
    return pd.DataFrame(BACKUP_RECIPES)

def get_backup_recipes():
    """Returns list of all backup recipes as dictionaries."""
    df = get_backup_df()
    return df.to_dict(orient="records")
