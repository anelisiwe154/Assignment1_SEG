**# Transformer-Based Question Answering System in Rust using Burn Framework \*\*Surname and Name:\*\* Anelisiwe Ntanjana**

**# Section 1: Introduction**

**##** Problem Statement and Motivation

Cape Peninsula University of Technology as a higher institution of learning is on structured documents such as calendars. However, extracting specific information from these documents manually can be inefficient and time-consuming. A Question Answering system enables users to query documents in natural language and automatically retrieve relevant answers. The goal of this project was to design and implement a complete machine learning pipeline in Rust using the Burn deep learning framework. The system reads calendar documents in \`.docx\` format, processes them into structured training data, trains a Transformer-based neural network, and allows users to ask questions via a command line interface.

****Frequently used test questions: ****

**-** When is GOOD FRIDAY?

- When is Start of Year for Academic Staff?

- When is University Employment Equity Forum? This project demonstrates the full machine learning lifecycle, including data processing, model design, training, checkpointing, and inference.

**### Overview of Approach The system consists of four major components:**

**-****Data Pipeline**: Loads \`.docx\` calendar files, extracts structured events, and converts them into training samples.

**- ****Model Architecture\*\*: Implements a Transformer encoder-based Q&A model using Burn.

**- ****Training Pipeline\*\*: Trains the model using labelled question-answer pairs and saves checkpoints.

**- ****Inference System\*\*: Allows users to ask questions via CLI and retrieves answers using retrieval-first and model-based fallback.

**### Key Design Decisions**

- Rust was used for performance, memory safety, and modern ML system design.

- Burn framework was chosen for modular architecture and GPU support.

- Transformer encoder selected for strong NLP performance.

- Retrieval-first inference implemented for higher accuracy.

- Modular and backend-agnostic design using Burn's generic Backend trait.

**# Section 2: Implementation**

**### Architecture Details**

**#### Model Architecture Overview**

**The model is a Transformer encoder designed for extractive question answering. \*\*Architecture flow: \*\***

1. Input Text

2. Token Embedding Layer

3. Positional Embedding Layer

4. Transformer Encoder (6 Layers)

5. Output Projection Layers

6. Start Logits and End Logits

**#### Layer Specifications**

**\| Component \| Dimension**

**\| \|\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--\|**

\| Vocabulary size \| 30,522 tokens

\| \| Maximum sequence length \| 128 tokens

\| \| Embedding dimension \| 256

\| \| Number of Transformer layers\| 6

\| \| Number of attention heads \| 8

\| \| Feedforward dimension \| 512

\| \| Dropout rate \| 0.1 \|

**#### Key Components Explanation**

- **Token Embeddings\*\*: Maps token IDs into dense vectors of dimension 256.

- **Positional Embeddings\*\*: Adds sequence order information.

- **Transformer Encoder Layers\*\*: Multi-head self-attention, feedforward network, layer normalization, residual connections.

- **Output Projection Layer\*\*: Produces start and end position logits representing the predicted answer span.

**### Data Pipeline**

- **Document Processing\*\*: Custom parser loads \`.docx\` files (e.g., \`calendar2024.docx\`, \`calendar2025.docx\`) and extracts events.

- Example: \*Event: GOOD FRIDAY, Date: 29 MARCH 2024\*

\- \*\*Training Data Generation\*\*: Events converted into Q&A pairs.

\- Example: \*Question: When is GOOD FRIDAY? → Answer: 29 MARCH 2024\*

\- \*\*Total training samples\*\*: 816

\*\*Tokenization Strategy\*\*: BERT-compatible tokenizer (\`bert-base-uncased-tokenizer.json\`). Input format: \`\[CLS\] question \[SEP\] context \[SEP\]\`

**\### Training Strategy**

\- \*\*Hyperparameters\*\*: - Learning rate: 0.0002 - Batch size: 8 - Epochs: 3 - Optimizer: Adam

\- Loss function: CrossEntropyLoss

\- \*\*Optimization Strategy\*\*: Forward pass > Loss computation > Backpropagation > Parameter update > Checkpoint saving. - \*\*Checkpoint file\*\*: \`checkpoints/model\`

**\### Challenges and Solutions**

\- Incorrect answer spans > Restricted prediction to tokens after \`\[SEP\]\`.

\- Irrelevant spans > Implemented retrieval-first inference.

\- Burn trait/backend issues > Used generic Backend trait.

\- Long documents > Sliding window context chunking.

**\## Section 3: Experiments and Results**

**\### Training Results**

\- Loss decreased over epochs:

\- Epoch 1: 1.80 - Epoch 2: 1.19 - Epoch 3: 1.05 - Training time: \~10--15 seconds on GPU.

**\### Model Performance**

**\*\*Example Q&A:\*\***

\- \*When is Start of Year for Academic Staff?* > **8 JANUARY 2024**

\- \*When is GOOD FRIDAY?\* > **08 APRIL 2024**

\- \*When is Dean and Directors?\* > \*\*22 JANUARY 2024\*\*

\- \*When is University Employment Equity Forum?\* > \*\*26 JANUARY 2024\*\*

\- \*When is New Year\'s Day?\* → \*\*1 JANUARY 2024

**\*\* \*\*Reasons for good performance:\*\***

\- Transformer learns contextual relationships.

\- Retrieval-first strategy improves accuracy.

\- Tokenizer ensures consistent input encoding.

**\### Failure Cases**

\- Questions do not present in calendar.

\- Ambiguous/incomplete questions.

\- Events spanning multiple contexts.

**\### Configuration Comparison**

\- \*\*Configuration A\*\*: d_model = 128

\- \*\*Configuration B\*\*: d_model = 256

- **Result\*\*: Configuration B achieved lower loss and better accuracy.

**\## Section 4: Conclusion**

**\### What Was Learned**

\- Transformer architecture
- ML pipeline implementation in Rust

- Burn framework usage

- Tokenization and NLP preprocessing

- Model training and inference design

**\### Challenges Encountered**

- Burn framework complexity

- Span prediction accuracy

- Dataset generation from documents

- Backend and tensor handling

**\### Potential Improvements**

- Larger datasets

- Pretrained model initialization

- Better validation metrics

- GPU optimization

- Improved span prediction scoring

**\### Future Work**

- Web interface

- Support for multiple document types

- Fine-tuning on larger corpora

- Deployment as API service

**## How to Run the System**

```bash cargo run
