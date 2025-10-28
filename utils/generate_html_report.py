#!/usr/bin/env python3
"""
HTML Report Generator for RAG Evaluation Results
===============================================

This script reads JSON evaluation results and generates a comprehensive HTML report
showing individual test results with questions, actual outputs, expected outputs,
and detailed evaluation metrics in a compact table format.

Key Features:
- Individual test analysis without summary averages
- Compact table format for metrics (Metric Name | Score)
- Color-coded scores for quick assessment
- Responsive design for mobile and desktop viewing
- Focus on detailed per-test evaluation rather than aggregate statistics

Usage:
    python generate_html_report.py                    # Uses latest evaluation file
    python generate_html_report.py results.json       # Specify JSON file
    python generate_html_report.py results.json report.html  # Custom output

The script automatically finds the most recent 'deepeval_rag_evaluation_with_*.json' file
if no JSON file is specified.

Report Structure:
- Header with evaluation details and total test count
- Individual test results showing:
  * Test question
  * Actual vs Expected outputs
  * Detailed metrics in table format:
    - RAG Contextual Metrics (Precision, Recall, Relevancy)
    - GEval Custom Metrics (Cultural Sensitivity, Historical Accuracy, etc.)
"""

import json
import sys
from pathlib import Path
import argparse
from datetime import datetime

def find_latest_evaluation_json(directory="."):
    """Find the most recent deepeval_rag_evaluation JSON file."""
    import glob

    # Look for files matching the pattern
    pattern = "deepeval_rag_evaluation_with_*.json"
    files = glob.glob(pattern)

    if not files:
        return None

    # Sort by timestamp in filename (newest first)
    def extract_timestamp(filename):
        # Extract timestamp from filename like "deepeval_rag_evaluation_with_20251028_143052.json"
        parts = filename.replace("deepeval_rag_evaluation_with_", "").replace(".json", "")
        return parts

    # Sort by timestamp (assuming format YYYYMMDD_HHMMSS)
    files.sort(key=extract_timestamp, reverse=True)
    return files[0]

def generate_html_report(json_file_path=None, output_file=None):
    """Generate HTML report from JSON evaluation results."""

    # If no JSON file specified, find the latest one
    if json_file_path is None:
        json_file_path = find_latest_evaluation_json()
        if json_file_path is None:
            print("❌ No evaluation JSON files found in current directory.")
            print("   Run the RAG evaluation script first to generate results.")
            return None
        print(f"📄 Using latest evaluation file: {json_file_path}")

    # Load JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get('detailed_results', [])

    if not results:
        print("No results found in JSON file")
        return

    # Generate timestamp for report
    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get evaluation model from first result
    evaluation_model = results[0].get('evaluation_model', 'Unknown') if results else 'Unknown'

    # Calculate summary statistics
    total_tests = len(results)

    # Extract all RAG scores
    precision_scores = [r['rag']['precision'] for r in results]
    recall_scores = [r['rag']['recall'] for r in results]
    relevancy_scores = [r['rag']['relevancy'] for r in results]

    # Extract all GEval scores
    cultural_scores = [r['geval']['cultural_sensitivity'] for r in results]
    historical_scores = [r['geval']['historical_accuracy'] for r in results]
    tourism_scores = [r['geval']['tourism_relevance'] for r in results]
    educational_scores = [r['geval']['educational_value'] for r in results]
    completeness_scores = [r['geval']['completeness'] for r in results]

    # Calculate averages
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
    avg_cultural = sum(cultural_scores) / len(cultural_scores)
    avg_historical = sum(historical_scores) / len(historical_scores)
    avg_tourism = sum(tourism_scores) / len(tourism_scores)
    avg_educational = sum(educational_scores) / len(educational_scores)
    avg_completeness = sum(completeness_scores) / len(completeness_scores)

    # Overall averages
    rag_avg = (avg_precision + avg_recall + avg_relevancy) / 3
    geval_avg = (avg_cultural + avg_historical + avg_tourism + avg_educational + avg_completeness) / 5
    overall_avg = (rag_avg + geval_avg) / 2

    # Count passed tests (score >= 0.7)
    def get_score_color(score):
        if score >= 0.8:
            return "#28a745"  # Green
        elif score >= 0.7:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red

    def format_score(score):
        return f"{score:.3f}"

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report - Detailed Test Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}

        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        .topic-highlight {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 600;
            font-size: 1.2em;
        }}

        .header p {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}

        .summary {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}

        .summary h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
            text-align: center;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .metric-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #3498db;
        }}

        .metric-card h3 {{
            color: #2c3e50;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .metric-score {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}

        .test-result {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}

        .test-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }}

        .test-title h2 {{
            margin: 0;
            font-size: 1.4em;
            font-weight: 600;
            color: white;
        }}

        .test-summary {{
            display: flex;
            gap: 15px;
        }}

        .summary-metrics {{
            display: flex;
            gap: 10px;
        }}

        .metric-badge {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 8px 15px;
            display: flex;
            align-items: center;
            gap: 8px;
            backdrop-filter: blur(10px);
        }}

        .metric-label {{
            font-size: 0.8em;
            font-weight: 600;
            opacity: 0.9;
        }}

        .metric-badge .metric-score {{
            font-size: 1.1em;
            font-weight: bold;
        }}

        .rag-badge {{
            background: rgba(52, 152, 219, 0.3);
        }}

        .geval-badge {{
            background: rgba(46, 204, 113, 0.3);
        }}

        .overall-badge {{
            background: rgba(155, 89, 182, 0.3);
        }}

        .test-content {{
            padding: 25px;
        }}

        .question-section {{
            margin-bottom: 25px;
        }}

        .question-section h3 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}

        .question {{
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            border-radius: 5px;
            font-weight: 500;
        }}

        .output-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}

        .output-box {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-top: 4px solid;
        }}

        .actual-output {{
            border-top-color: #28a745;
        }}

        .expected-output {{
            border-top-color: #dc3545;
        }}

        .output-box h4 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .output-text {{
            line-height: 1.6;
            color: #555;
        }}

        .metrics-detailed {{
            display: flex;
            flex-direction: column;
            gap: 25px;
        }}

        .metrics-group {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
        }}

        .metrics-group h5 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.1em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }}

        .metric-item {{
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }}

        .metric-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }}

        .metric-name {{
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            font-weight: 600;
        }}

        .metric-value {{
            font-size: 1.6em;
            font-weight: bold;
            margin-bottom: 8px;
        }}

        .metric-desc {{
            font-size: 0.8em;
            color: #95a5a6;
            line-height: 1.3;
            font-style: italic;
        }}

        .metrics-table-container {{
            overflow-x: auto;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
        }}

        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .metrics-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.9em;
        }}

        .metrics-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
            font-size: 0.9em;
        }}

        .metrics-table tbody tr:hover {{
            background: #f8f9fa;
        }}

        .metric-group-header {{
            background: #e9ecef !important;
            font-weight: bold;
            color: #2c3e50;
        }}

        .metric-group-header td {{
            padding: 15px;
            font-size: 1em;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .output-section {{
                grid-template-columns: 1fr;
            }}

            .test-header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 15px;
            }}

            .summary-metrics {{
                width: 100%;
                justify-content: center;
            }}

            .test-title h2 {{
                font-size: 1.2em;
            }}

            .metrics-detailed {{
                gap: 20px;
            }}

            .metrics-group {{
                padding: 15px;
            }}

            .header h1 {{
                font-size: 2em;
            }}

            .metrics-table {{
                font-size: 0.8em;
            }}

            .metrics-table th,
            .metrics-table td {{
                padding: 8px 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RAG Evaluation Report</h1>
            <p class="topic-highlight">Detailed Individual Test Analysis</p>
            <p><strong>Evaluation Model:</strong> {evaluation_model}</p>
            <p>Report Generated: {report_timestamp}</p>
            <p><strong>Total Tests:</strong> {total_tests}</p>
        </div>
"""

    # Add individual test results
    for i, result in enumerate(results, 1):
        html_content += f"""
        <div class="test-result">
            <div class="test-header">
                <div class="test-title">
                    <h2>Test {i}: {result['query'][:60]}{"..." if len(result['query']) > 60 else ""}</h2>
                </div>
            </div>
            <div class="test-content">
                <div class="question-section">
                    <h3>❓ Question</h3>
                    <div class="question">{result['query']}</div>
                </div>

                <div class="output-section">
                    <div class="output-box actual-output">
                        <h4>📝 Actual Output</h4>
                        <div class="output-text">{result['actual_output']}</div>
                    </div>
                    <div class="output-box expected-output">
                        <h4>🎯 Expected Output</h4>
                        <div class="output-text">{result['expected_output']}</div>
                    </div>
                </div>

                <div class="metrics-section">
                    <h4>📊 Detailed Metrics</h4>
                    <div class="metrics-table-container">
                        <table class="metrics-table">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Score</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr class="metric-group-header">
                                    <td colspan="2"><strong>RAG Contextual Metrics</strong></td>
                                </tr>
                                <tr>
                                    <td>Precision</td>
                                    <td style="color: {get_score_color(result['rag']['precision'])}">{format_score(result['rag']['precision'])}</td>
                                </tr>
                                <tr>
                                    <td>Recall</td>
                                    <td style="color: {get_score_color(result['rag']['recall'])}">{format_score(result['rag']['recall'])}</td>
                                </tr>
                                <tr>
                                    <td>Relevancy</td>
                                    <td style="color: {get_score_color(result['rag']['relevancy'])}">{format_score(result['rag']['relevancy'])}</td>
                                </tr>
                                <tr class="metric-group-header">
                                    <td colspan="2"><strong>GEval Custom Metrics</strong></td>
                                </tr>
                                <tr>
                                    <td>Cultural Sensitivity</td>
                                    <td style="color: {get_score_color(result['geval']['cultural_sensitivity'])}">{format_score(result['geval']['cultural_sensitivity'])}</td>
                                </tr>
                                <tr>
                                    <td>Historical Accuracy</td>
                                    <td style="color: {get_score_color(result['geval']['historical_accuracy'])}">{format_score(result['geval']['historical_accuracy'])}</td>
                                </tr>
                                <tr>
                                    <td>Tourism Relevance</td>
                                    <td style="color: {get_score_color(result['geval']['tourism_relevance'])}">{format_score(result['geval']['tourism_relevance'])}</td>
                                </tr>
                                <tr>
                                    <td>Educational Value</td>
                                    <td style="color: {get_score_color(result['geval']['educational_value'])}">{format_score(result['geval']['educational_value'])}</td>
                                </tr>
                                <tr>
                                    <td>Completeness</td>
                                    <td style="color: {get_score_color(result['geval']['completeness'])}">{format_score(result['geval']['completeness'])}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
"""

    html_content += """
    </div>

    <div class="footer">
        <p>Generated by RAG Evaluation Framework</p>
    </div>
</body>
</html>
"""

    # Determine output file name
    if output_file is None:
        json_stem = Path(json_file_path).stem
        output_file = f"{json_stem}_report.html"

    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML report generated: {output_file}")
    print(f"📊 Report includes {total_tests} test results with detailed metrics")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate HTML report from RAG evaluation JSON results')
    parser.add_argument('json_file', nargs='?', help='Path to the JSON results file (optional - uses latest if not specified)')
    parser.add_argument('output_file', nargs='?', help='Optional output HTML file name')

    args = parser.parse_args()

    # If JSON file not specified, find_latest_evaluation_json will be called in generate_html_report
    json_file_path = args.json_file

    try:
        output_file = generate_html_report(json_file_path, args.output_file)
        if output_file:
            print(f"\n🎉 Success! Open '{output_file}' in your web browser to view the report.")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()