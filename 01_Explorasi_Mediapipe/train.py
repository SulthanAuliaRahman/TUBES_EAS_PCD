import streamlit as st
import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

def display_train_page():
    # Streamlit session state initialization
    if "data" not in st.session_state:
        st.session_state.data = []
    if "input_path" not in st.session_state:
        st.session_state.input_path = ""
    if "is_folder" not in st.session_state:
        st.session_state.is_folder = False
    if "training_status" not in st.session_state:
        st.session_state.training_status = False

    # File and folder upload widgets
    st.subheader("Pilih Data untuk Pelatihan")
    uploaded_file = st.file_uploader("Pilih File JSON", type=["json"])
    folder_path = st.text_input("Masukkan Path Folder JSON (opsional)", "")
    train_button = st.button("Mulai Pelatihan", disabled=st.session_state.training_status)

    # Output area
    output_container = st.container()

    # Handle file upload
    if uploaded_file is not None and st.session_state.input_path != uploaded_file.name:
        try:
            st.session_state.data = json.load(uploaded_file)
            st.session_state.input_path = uploaded_file.name
            st.session_state.is_folder = False
            output_container.write(f"File dipilih: {uploaded_file.name}")
            output_container.write(f"Memuat {len(st.session_state.data)} sampel dari file.")
        except json.JSONDecodeError:
            output_container.error("File JSON tidak valid.")
            st.session_state.data = []

    # Handle folder path
    if folder_path and st.session_state.input_path != folder_path:
        try:
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            if not json_files:
                output_container.error("Tidak ada file JSON di dalam folder.")
                st.session_state.data = []
            else:
                st.session_state.data = []
                for json_file in json_files:
                    with open(os.path.join(folder_path, json_file), 'r') as f:
                        file_data = json.load(f)
                        st.session_state.data.extend(file_data)
                st.session_state.input_path = folder_path
                st.session_state.is_folder = True
                output_container.write(f"Folder dipilih: {folder_path}")
                output_container.write(f"Memuat {len(st.session_state.data)} sampel dari {len(json_files)} file.")
        except FileNotFoundError:
            output_container.error("Folder tidak ditemukan.")
            st.session_state.data = []
        except json.JSONDecodeError:
            output_container.error("Ada file JSON yang tidak valid di dalam folder.")
            st.session_state.data = []

    # Training logic
    if train_button and st.session_state.data:
        st.session_state.training_status = True
        output_container.write("Memulai pelatihan dengan 5-fold cross-validation...")

        # Step 1: Extract features and labels
        features = []
        labels = []
        expected_num_landmarks = len(st.session_state.data[0]['landmarks'])
        for sample in st.session_state.data:
            try:
                gesture = sample['name']
                landmarks = sample['landmarks']
                if len(landmarks) != expected_num_landmarks:
                    output_container.warning(f"Sampel memiliki jumlah landmark yang salah: {len(landmarks)} (diharapkan {expected_num_landmarks})")
                    continue
                landmark_coords = []
                for landmark in landmarks:
                    landmark_coords.extend([landmark['x'], landmark['y'], landmark['z']])
                features.append(landmark_coords)
                labels.append(gesture)
            except KeyError:
                output_container.warning("Ada kunci yang hilang di dalam sampel.")
                continue

        if not features:
            output_container.error("Tidak ada data valid untuk dilatih.")
            st.session_state.training_status = False
            return

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        output_container.write(f"Bentuk fitur: {X.shape}")
        output_container.write(f"Jumlah label: {len(y)}")

        # Step 2: Perform 5-fold cross-validation
        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_reports = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model for this fold
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)
            report = classification_report(y_test, y_pred, target_names=sorted(set(y)), output_dict=True)
            fold_reports.append((fold, accuracy, report))
            output_container.write(f"Fold {fold} - Akurasi: {accuracy:.2f}")

        # Calculate mean and standard deviation of accuracies
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        output_container.write(f"\nAkurasi Rata-rata: {mean_accuracy:.2f} ± {std_accuracy:.2f}")

        # Display classification report for each fold
        output_container.subheader("Laporan Klasifikasi per Fold")
        for fold, accuracy, report in fold_reports:
            output_container.write(f"\nFold {fold} (Akurasi: {accuracy:.2f}):")
            for gesture, metrics in report.items():
                if isinstance(metrics, dict):  # Only display per-class metrics
                    output_container.write(f"  {gesture}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-Score={metrics['f1-score']:.2f}")

        # Step 3: Visualize accuracy per fold
        output_container.subheader("Akurasi per Fold")
        chart_data = pd.DataFrame({
            "Fold": [f"Fold {i}" for i in range(1, k + 1)],
            "Akurasi": fold_accuracies
        })
        st.bar_chart(chart_data.set_index("Fold"))

        # Step 4: Train final model on all data
        final_model = RandomForestClassifier(n_estimators=100, random_state=42)
        final_model.fit(X, y)
        output_container.write("\nMelatih model akhir pada seluruh data...")

        # Step 5: Save the final model
        joblib.dump(final_model, 'gesture_classifier_cv.pkl')
        output_container.success("Model akhir disimpan sebagai 'gesture_classifier_cv.pkl'")
        st.session_state.training_status = False