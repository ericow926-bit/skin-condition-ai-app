# Model: EfficientNet-B0, frozen backbone, trained on DermNet
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from huggingface_hub import hf_hub_download
from PIL import Image

CLASS_NAMES = [
    "Eczema",
    "Hives",
    "Acne/Rosacea",
    "Psoriasis/Lichen Planus",
    "Contact Dermatitis/Poison Ivy",
    "Ringworm/Fungal Infections",
]

# Contact Dermatitis / Poison Ivy flagged as lower confidence according to findings
LOWER_CONFIDENCE_CLASSES = {"Contact Dermatitis/Poison Ivy"}
HF_MODEL_REPO = "ericc926/skin-condition-classifier"
HF_MODEL_FILENAME = "best_model_stage1.pt"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Educational Content to be displayed to the user after answering questions
EDUCATIONAL_CONTENT = {
    "Eczema": {
        "what_it_is": (
            "Eczema (atopic dermatitis) is a common condition that causes patches "
            "of dry, itchy, inflamed skin. It often appears in the folds of elbows "
            "and knees, although it can show up almost anywhere."
        ),
        "common_causes": [
            "A tendency to run in families alongside asthma or allergies",
            "A skin barrier that lets moisture out and irritants in more easily than typical skin",
            "Triggers like dry weather, harsh soaps, stress, or certain fabrics",
        ],
        "self_care": [
            "Moisturize generously, ideally right after bathing while skin is still damp",
            "Use fragrance free, gentle cleansers instead of harsh soaps",
            "Avoid scratching where possible as it can worsen irritation and risk infection",
            "Identify and avoid personal triggers (certain fabrics, detergents, etc.)",
        ],
    },
    "Hives": {
        "what_it_is": (
            "Hives (urticaria) are raised, itchy welts that can appear suddenly and "
            "often shift location over hours. They're usually a short-term allergic "
            "or irritant response."
        ),
        "common_causes": [
            "Allergic reactions to food, medication, or insect bites",
            "Physical triggers like heat, cold, or pressure on the skin",
            "Infections or, sometimes, stress",
        ],
        "self_care": [
            "Avoid known triggers if you can identify one",
            "Cool compresses can help ease itching and swelling",
            "Loose, breathable clothing reduces skin irritation",
            "Seek urgent care if hives come with swelling of the face/throat or trouble "
            "breathing which can signal a serious allergic reaction",
        ],
    },
    "Acne/Rosacea": {
        "what_it_is": (
            "This category covers acne (clogged pores causing pimples, blackheads, "
            "or cysts) and rosacea (facial redness and visible blood vessels, "
            "sometimes with bumps). Two distinct conditions grouped together here "
            "because of how the source dataset is organized."
        ),
        "common_causes": [
            "Acne: excess oil production, clogged pores, hormonal changes, certain bacteria",
            "Rosacea: not fully understood, but often linked to blood vessel sensitivity "
            "and triggers like sun, heat, or spicy food",
        ],
        "self_care": [
            "Use gentle, non-comedogenic (less likely to clog pores) skincare products",
            "Avoid picking or popping as it can worsen scarring",
            "For rosacea, track and avoid personal flare triggers (sun exposure, alcohol, "
            "spicy food are common ones)",
            "Consistent, gentle routines tend to help more than aggressive treatment",
        ],
    },
    "Psoriasis/Lichen Planus": {
        "what_it_is": (
            "This category covers psoriasis (an immune-driven condition causing "
            "thick, scaly patches) and lichen planus (an inflammatory condition "
            "causing itchy, flat-topped bumps), they are grouped together here based on "
            "the source dataset's original categorization."
        ),
        "common_causes": [
            "Psoriasis: an overactive immune response that speeds up skin cell turnover",
            "Lichen planus: also immune-related, sometimes linked to certain medications "
            "or hepatitis C",
            "Both can be triggered or worsened by stress",
        ],
        "self_care": [
            "Moisturize regularly to reduce scaling and discomfort",
            "Avoid scratching or picking at patches",
            "Track flare patterns: stress, weather, and certain medications are common triggers",
            "These conditions often benefit from professional treatment beyond self-care alone",
        ],
    },
    "Contact Dermatitis/Poison Ivy": {
        "what_it_is": (
            "Contact dermatitis is skin irritation or an allergic reaction from "
            "something touching the skin, ranging from plants like poison ivy to "
            "soaps, metals, or cosmetics."
        ),
        "common_causes": [
            "Direct contact with an irritant (harsh soap, certain chemicals)",
            "Allergic reaction to a specific substance (nickel, certain plants, fragrances)",
            "Reactions often appear in the shape/pattern of contact with the trigger",
        ],
        "self_care": [
            "Identify and avoid the suspected trigger going forward",
            "Wash the area gently to remove any remaining irritant",
            "Cool compresses and fragrance free moisturizers can ease discomfort",
            "If a new product, plant, or metal was involved recently, that's a strong "
            "clue worth mentioning to a doctor",
        ],
    },
    "Ringworm/Fungal Infections": {
        "what_it_is": (
            "Despite the name, ringworm isn't a worm, it's actually a fungal infection "
            "that often (but not always) causes a ring-shaped, scaly, itchy patch."
        ),
        "common_causes": [
            "Contact with fungi from skin-to-skin contact, contaminated surfaces, or animals",
            "Warm, moist environments (shared gym equipment, locker rooms) are common "
            "transmission points",
        ],
        "self_care": [
            "Keep the area clean and dry as fungi thrive in moisture",
            "Avoid sharing towels, clothing, or equipment while it's active",
            "Over the counter antifungal creams are the standard first self-care option",
            "See a doctor if it doesn't improve within a couple of weeks of consistent "
            "antifungal use",
        ],
    },
}

DURATION_OPTIONS = ["A few days", "A few weeks", "Months or longer"]
ITCH_OPTIONS = ["None", "Mild", "Severe"]
LOCATION_OPTIONS = ["Face", "Trunk (chest/back/stomach)", "Limbs (arms/legs)", "Hands or feet", "Other"]
EXPOSURE_OPTIONS = ["No", "Yes"]

def build_contextual_notes(predicted_class, duration, itch, location, exposure):
    notes = []

    if duration == "Months or longer":
        notes.append(
            "Because this has persisted for months or longer, it's worth prioritizing "
            "an in-person evaluation soon, regardless of the result above. Persistent "
            "skin changes are generally worth a professional look."
        )

    if exposure == "Yes":
        notes.append(
            "You mentioned a recent new exposure (product, plant, metal, etc.). That's "
            "something specifically worth mentioning to a doctor, since it's a strong "
            "clue for contact dermatitis regardless of what this tool predicted."
        )

    itchy_conditions = {"Eczema", "Hives", "Contact Dermatitis/Poison Ivy", "Ringworm/Fungal Infections"}
    if itch == "Severe" and predicted_class not in itchy_conditions:
        notes.append(
            "Severe itching isn't the most typical presentation for the predicted "
            "category so this doesn't rule it out, but it's a detail worth mentioning "
            "if you do see a professional."
        )

    return notes


# Model loading:
@st.cache_resource
def load_model():
    # Downloads the checkpoint from Hugging Face Hub once per app session
    weights_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))

    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def predict(model, image: Image.Image):
    img_tensor = eval_transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    return probs.tolist()


# App layout:
st.set_page_config(page_title="Skin Condition Visual Similarity", page_icon="", layout="centered")

st.title("AI Based Skin Condition Visual Similarity Tool")

st.warning(
    "**This is an educational tool, not a medical diagnosis.** It compares your "
    "photo's visual patterns to a training dataset and reports similarity, it "
    "cannot confirm what a skin condition actually is. Always consult a healthcare "
    "professional for an actual diagnosis, especially for anything persistent, "
    "spreading, or concerning."
)

with st.expander("About this tool and its limitations"):
    st.markdown(
        "- Trained on the DermNet dataset (consumer/educational photos, not "
        "clinically verified labels)\n"
        "- Recognizes 6 categories, several of which are merged groupings from "
        "the source dataset (e.g. results labeled 'Acne/Rosacea' do not "
        "distinguish between the two)\n"
        "- Test-set macro-F1: 0.520, a moderate honest accuracy level, "
        "not a clinical-grade result\n"
        "- Contact Dermatitis/Poison Ivy is a known weaker category (will be flagged "
        "below whenever it's the top result)"
    )

st.subheader("1. Upload a photo")
uploaded_file = st.file_uploader(
    "Upload a clear, well-lit photo of the affected skin area", type=["jpg", "jpeg", "png"]
)

st.subheader("2. A few quick questions")
st.caption(
    "These don't affect the prediction, they can only affect which notes and "
    "self-care tips are shown alongside it."
)

col1, col2 = st.columns(2)
with col1:
    duration = st.selectbox("How long has this been present?", DURATION_OPTIONS)
    location = st.selectbox("Where on the body is it?", LOCATION_OPTIONS)
with col2:
    itch = st.selectbox("Itch level?", ITCH_OPTIONS)
    exposure = st.selectbox("Any recent new exposure (soap, lotion, plant, metal, etc.)?", EXPOSURE_OPTIONS)

submitted = st.button("Analyze", type="primary", disabled=uploaded_file is None)

if submitted and uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("3. Result")
    result_col, image_col = st.columns([3, 2])

    with image_col:
        st.image(image, caption="Uploaded photo", use_container_width=True)

    with st.spinner("Analyzing..."):
        model = load_model()
        probs = predict(model, image)

    ranked = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)
    top_class, top_prob = ranked[0]

    with result_col:
        st.markdown(f"### Visual similarity to: **{top_class}**")
        st.markdown(f"Confidence: **{top_prob * 100:.1f}%**")

        if top_class in LOWER_CONFIDENCE_CLASSES:
            st.error(
                "**Lower-confidence category.** Across repeated testing, this "
                "category has consistently underperformed the others (test F1 "
                "around 0.32–0.38 across five separate measurements), likely due to "
                "visual overlap with other conditions and a smaller training set. "
                "Treat this result with extra caution and consider a professional "
                "opinion more heavily than usual."
            )

        st.markdown("**Full breakdown:**")
        st.bar_chart({name: prob for name, prob in ranked})

    for note in build_contextual_notes(top_class, duration, itch, location, exposure):
        st.info(note)

    st.subheader("4. About this condition")
    content = EDUCATIONAL_CONTENT[top_class]
    st.markdown(f"**What it is:** {content['what_it_is']}")

    st.markdown("**Common causes/triggers:**")
    for cause in content["common_causes"]:
        st.markdown(f"- {cause}")

    st.markdown("**General self-care suggestions:**")
    for tip in content["self_care"]:
        st.markdown(f"- {tip}")

    st.info(
        "This is general educational information, not personalized medical advice. "
        "If symptoms are severe, spreading, or not improving, please see a doctor "
        "or dermatologist."
    )
elif submitted and uploaded_file is None:
    st.error("Please upload a photo first.")
