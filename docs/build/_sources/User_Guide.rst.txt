=================
User Guide
=================

This User Guide will walk you through how to use the CKD prediction model.

1. **Data Input:**
   
   The model requires patient data in a specific format. Ensure your data matches the format used in the Proinsalud dataset, which the model was trained on.

2. **Running the Model:**

   Once your data is prepared, you can use the model to predict CKD risk. Call the `predict` function in `ckd_model.py` with your data as the argument.

3. **Interpreting Results:**

   The output of the model is a risk classification for CKD. This will help you to identify the patients at risk and take necessary actions.

Remember, this model is a tool to assist with CKD risk prediction and should not replace professional medical advice.
