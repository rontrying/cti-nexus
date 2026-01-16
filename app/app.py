import argparse
import json
import os
import sys
import traceback

from graph_constructor import create_graph_visualization
from utils.gradio_utils import build_interface, run_pipeline
from utils.http_server_utils import setup_http_server
import gradio as gr
import os
from pypdf import PdfReader
from app.cti_processor import CTIProcessor
from app.graph_constructor import GraphConstructor
from utils.model_utils import (
    MODELS,
    check_api_key,
)
from utils.path_utils import resolve_path


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="CTINexus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--text", "-t",
        type=str,
        help="Input threat intelligence text to process"
    )
    input_group.add_argument(
        "--input-file", "-i",
        type=str,
        help="Path to file containing threat intelligence text"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="AI provider to use: OpenAI, Gemini, AWS, or Ollama (auto-detected if not specified)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for all text processing steps (e.g., gpt-4o, o4-mini)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model for entity alignment (e.g., text-embedding-3-large)"
    )
    parser.add_argument(
        "--ie-model",
        type=str,
        help="Override model for Intelligence Extraction"
    )
    parser.add_argument(
        "--et-model", 
        type=str,
        help="Override model for Entity Tagging"
    )
    parser.add_argument(
        "--ea-model",
        type=str, 
        help="Override embedding model for Entity Alignment"
    )
    parser.add_argument(
        "--lp-model",
        type=str,
        help="Override model for Link Prediction"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for entity alignment (0.0-1.0, default: 0.6)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (if not specified, saves to app/output/ directory)"
    )
    
    return parser


def get_default_models_for_provider(provider):
    defaults = {
        "OpenAI": {
            "model": "o4-mini",
            "embedding_model": "text-embedding-3-large"
        },
        "Gemini": {
            "model": "gemini-2.0-flash",
            "embedding_model": "gemini-embedding-001"
        },
        "AWS": {
            "model": "anthropic.claude-3-5-sonnet",
            "embedding_model": "amazon.titan-embed-text-v2:0"
        },
        "Ollama": {
            "model": "llama3.1:8b",
            "embedding_model": "nomic-embed-text"
        }
    }
    return defaults.get(provider, {})


def run_cmd_pipeline(args):
    if args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except FileNotFoundError:
            print(f"Error: Input file '{args.input_file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)
    else:
        text = args.text
    
    if not text:
        print("Error: No input text provided")
        sys.exit(1)
    
    provider = args.provider
    available_providers = list(MODELS.keys())

    if provider:
        provider_matched = next((p for p in available_providers if provider.lower() == p.lower()), None)
        if not provider_matched:
            print(f"Error: Provider '{provider}' not available. Available providers: {available_providers}")
            sys.exit(1)
        provider = provider_matched
    else:
        # Auto-detect based on available API keys
        if available_providers:
            provider = available_providers[0]
        else:
            print("Error: No API keys configured")
            sys.exit(1)
    
    defaults = get_default_models_for_provider(provider)

    # Set models with fallbacks to defaults
    base_model = args.model or defaults.get("model")
    base_embedding_model = args.embedding_model or defaults.get("embedding_model")
    
    ie_model = f"{provider}/{args.ie_model or base_model}"
    et_model = f"{provider}/{args.et_model or base_model}"
    ea_model = f"{provider}/{args.ea_model or base_embedding_model}"
    lp_model = f"{provider}/{args.lp_model or base_model}"

    print(f"Running CTINexus with {provider} provider...")
    print(f"IE: {ie_model}, ET: {et_model}, EA: {ea_model}, LP: {lp_model}")
    
    try:
        result = run_pipeline(
            text=text,
            ie_model=ie_model,
            et_model=et_model, 
            ea_model=ea_model,
            lp_model=lp_model,
            similarity_threshold=args.similarity_threshold
        )
        
        if result.startswith("Error:"):
            print(result)
            sys.exit(1)

        # Determine output file
        if args.output:
            output_file = args.output
        elif args.input_file:
            # Use input filename with _output.json
            input_basename = os.path.basename(args.input_file)
            base_name = os.path.splitext(input_basename)[0]
            output_file = resolve_path("output", f"{base_name}_output.json")
        else:
            output_file = resolve_path("output", "output.json")
        
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
            
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Results written to: {output_file}")
        except Exception as e:
            print(f"Error writing output file: {e}")
            print(result)
            sys.exit(1)

        # Create Entity Relation Graph
        result_dict = json.loads(result)
        _, filepath = create_graph_visualization(result_dict)
        print(f"Entity Relation Graph: {filepath}")

        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    api_keys_available = check_api_key()

    run_gui = not args.text and not args.input_file
    
    if run_gui:
        # GUI mode
        warning = None
        if not api_keys_available:
            warning = "⚠️   Warning: No API Keys Configured. Please provide one API key in the `.env` file from the supported providers.\n"
            print(warning)
        build_interface(warning)
    else:
        # Command line mode
        if not api_keys_available:
            print("⚠️   Warning: No API Keys Configured. Please provide one API key in the `.env` file from the supported providers.\n")
            sys.exit(1)
        
        run_cmd_pipeline(args)


if __name__ == "__main__":
    # HTTP server to serve pyvis files
    setup_http_server()

    main()

# Inisialisasi Modul
processor = CTIProcessor()
graph_constructor = GraphConstructor()

def extract_text_from_pdf(file_obj):
    """Fungsi helper untuk membaca teks dari PDF"""
    try:
        reader = PdfReader(file_obj.name)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def process_input(input_text, input_file, model_name):
    # 1. Cek apakah user upload file. Jika ya, prioritas pakai isi file.
    if input_file is not None:
        extracted_text = extract_text_from_pdf(input_file)
        # Jika teks terlalu panjang, ambil 4000 karakter pertama (agar tidak error token limit)
        input_text = extracted_text[:8000] if len(extracted_text) > 8000 else extracted_text

    if not input_text:
        return "Mohon masukkan teks atau upload PDF.", None

    # 2. Proses Ekstraksi Triplet menggunakan LLM
    # (Pastikan CTIProcessor di cti_processor.py Anda mengembalikan list triplet)
    triplets = processor.process(input_text, model_name)
    
    # 3. Konstruksi Graf
    graph_constructor.add_triplets(triplets)
    
    # 4. Generate Visualisasi HTML
    html_graph = graph_constructor.generate_interactive_graph()
    
    # Kembalikan string JSON (untuk tab Text) dan HTML (untuk tab Visualisasi)
    import json
    json_output = json.dumps(triplets, indent=2)
    
    if html_graph is None:
        html_graph = "<div style='color:white'>Tidak ada relasi yang ditemukan untuk divisualisasikan.</div>"
        
    return json_output, html_graph

# --- Setup UI Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ CTINexus: Automated Threat Intelligence Graph")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input Area
            input_text = gr.Textbox(lines=10, label="Input CTI Report (Text)", placeholder="Paste report here...")
            input_file = gr.File(label="Or Upload CTI Report (PDF)", file_types=[".pdf"])
            
            # Model Selection (Sesuaikan dengan config Anda)
            model_dropdown = gr.Dropdown(
                choices=["gpt-3.5-turbo", "gpt-4", "local-model"], 
                value="gpt-3.5-turbo", 
                label="Select Model"
            )
            
            submit_btn = gr.Button("🚀 Extract Knowledge Graph", variant="primary")
            
        with gr.Column(scale=2):
            # Output Area dengan Tabs
            with gr.Tabs():
                with gr.TabItem("🕸️ Interactive Graph"):
                    # Komponen HTML untuk render pyvis
                    graph_output = gr.HTML(label="Knowledge Graph Visualization")
                with gr.TabItem("📄 JSON Triples"):
                    json_output = gr.Code(language="json", label="Extracted Entities & Relations")

    # Event Handler
    submit_btn.click(
        fn=process_input,
        inputs=[input_text, input_file, model_dropdown],
        outputs=[json_output, graph_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)