import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def load_fasta(file_path):
    sequences = []
    seq = ""
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.upper()
        if seq:
            sequences.append(seq)
    return sequences
#Zadržava samo sekvence dužine 50-3000 bp koje sadrže isključivo A, T, G, C
def clean_sequences(sequences, min_len=50, max_len=3000):
    return [s for s in sequences if min_len <= len(s) <= max_len and set(s) <= set("ATGC")]
#Broji pojavljivanje svakog k-mera u jednoj sekvenci
def kmer_frequency(sequence, k):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)
#Pravi matricu X gdje su redovi sekvence, a kolone svi pronađeni k-meri
def build_kmer_matrix(sequences, k):
    all_kmers = set()
    kmer_counts = []
    for seq in sequences:
        counts = kmer_frequency(seq, k)
        kmer_counts.append(counts)
        all_kmers.update(counts.keys())
    all_kmers = sorted(all_kmers)
    X = np.zeros((len(sequences), len(all_kmers)))
    for i, counts in enumerate(kmer_counts):
        for j, kmer in enumerate(all_kmers):
            X[i, j] = counts.get(kmer, 0)
    return X, all_kmers

DATA_DIR = "data"
sequences = []
labels = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".fasta") or filename.endswith(".fa"):
        species_name = filename.replace(".fasta", "").replace(".fa", "")
        file_path = os.path.join(DATA_DIR, filename)
        seqs = load_fasta(file_path)
        seqs = clean_sequences(seqs)
        sequences.extend(seqs)
        labels.extend([species_name] * len(seqs))
        print(f"{filename}: {len(seqs)} sekvenci")

print(f"\nUkupno: {len(sequences)} sekvenci iz {len(set(labels))} vrsta")
#K-mer kodiranje za k = 3, 4, 5, 6 i klasifikacija logističkom regresijom
for k in [3, 4, 5, 6]:
    print(f"\n--- k = {k} ---")
    X, kmers = build_kmer_matrix(sequences, k)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=lab, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA of Frog DNA k-mer Features (k={k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pca_k{k}.png") #Slika za izvještaj
    plt.show()

    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels])
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y   #80/20 podjela, stratifikovano po klasama
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=unique_labels))
    print("Accuracy:", accuracy_score(y_test, y_pred))
#Zadatak 2: ista analiza ali samo sa dvije vrste
print("\n--- Zadatak 2: dvije vrste ---")
unique_labels = sorted(set(labels))
species_to_remove = unique_labels[0]
mask = [l != species_to_remove for l in labels]
sequences_2 = [seq for seq, m in zip(sequences, mask) if m]
labels_2 = [l for l, m in zip(labels, mask) if m]

X2, _ = build_kmer_matrix(sequences_2, 4)
X2_scaled = StandardScaler().fit_transform(X2) #svodim sve k-mer frekvencije na istu skalu
X2_pca = PCA(n_components=2).fit_transform(X2_scaled)

plt.figure(figsize=(8, 6))
for lab in sorted(set(labels_2)):
    idx = [i for i, l in enumerate(labels_2) if l == lab]
    plt.scatter(X2_pca[idx, 0], X2_pca[idx, 1], label=lab, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Frog DNA (2 species, k=4)")
plt.legend()
plt.tight_layout()
plt.savefig("pca_2species.png")
plt.show()

label_map2 = {lab: i for i, lab in enumerate(sorted(set(labels_2)))}
y2 = np.array([label_map2[l] for l in labels_2])
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_scaled, y2, test_size=0.2, random_state=42, stratify=y2
)
model2 = LogisticRegression(max_iter=1000)
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)
print(classification_report(y2_test, y2_pred, target_names=sorted(set(labels_2))))
print("Accuracy (2 species):", accuracy_score(y2_test, y2_pred))
#Zadatak 3: poređenje SVM i Random Forest modela
print("\n--- Zadatak 3: SVM i Random Forest ---")
unique_labels = sorted(set(labels))
label_map = {lab: i for i, lab in enumerate(unique_labels)}
y = np.array([label_map[l] for l in labels])
X, _ = build_kmer_matrix(sequences, 4)
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\nSVM:")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(classification_report(y_test, y_pred_svm, target_names=unique_labels))
print("Accuracy SVM:", accuracy_score(y_test, y_pred_svm))

print("\nRandom Forest:")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf, target_names=unique_labels))
print("Accuracy RF:", accuracy_score(y_test, y_pred_rf))