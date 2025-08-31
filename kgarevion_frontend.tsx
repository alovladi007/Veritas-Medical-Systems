import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Send, 
  Loader2, 
  Brain, 
  Network, 
  CheckCircle, 
  XCircle,
  Activity,
  ChevronDown,
  User,
  LogOut,
  History,
  Settings
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

// Custom hook for WebSocket connection
const useWebSocket = (url) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const ws = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket(url);
    
    ws.current.onopen = () => setIsConnected(true);
    ws.current.onclose = () => setIsConnected(false);
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLastMessage(data);
    };

    return () => {
      ws.current.close();
    };
  }, [url]);

  const sendMessage = useCallback((message) => {
    if (ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    }
  }, []);

  return { isConnected, lastMessage, sendMessage };
};

// Triplet Visualization Component
const TripletGraph = ({ triplets }) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!triplets || triplets.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate node positions
    const nodes = new Map();
    const edges = [];
    
    triplets.forEach((triplet, index) => {
      const angle = (index * 2 * Math.PI) / triplets.length;
      const radius = 150;
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      
      if (!nodes.has(triplet.head)) {
        nodes.set(triplet.head, {
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
          label: triplet.head,
          type: 'entity'
        });
      }
      
      if (!nodes.has(triplet.tail)) {
        nodes.set(triplet.tail, {
          x: centerX + radius * Math.cos(angle + Math.PI),
          y: centerY + radius * Math.sin(angle + Math.PI),
          label: triplet.tail,
          type: 'entity'
        });
      }
      
      edges.push({
        from: triplet.head,
        to: triplet.tail,
        label: triplet.relation
      });
    });
    
    // Draw edges
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2;
    edges.forEach(edge => {
      const from = nodes.get(edge.from);
      const to = nodes.get(edge.to);
      
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
      
      // Draw relation label
      ctx.fillStyle = '#64748b';
      ctx.font = '12px sans-serif';
      ctx.fillText(
        edge.label,
        (from.x + to.x) / 2,
        (from.y + to.y) / 2
      );
    });
    
    // Draw nodes
    nodes.forEach(node => {
      ctx.beginPath();
      ctx.arc(node.x, node.y, 20, 0, 2 * Math.PI);
      ctx.fillStyle = node.type === 'disease' ? '#ef4444' : '#3b82f6';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw labels
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.label.substring(0, 5), node.x, node.y);
    });
  }, [triplets]);
  
  return (
    <div className="border rounded-lg p-4 bg-gray-50">
      <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
        <Network className="w-4 h-4" />
        Knowledge Graph Visualization
      </h3>
      <canvas 
        ref={canvasRef}
        width={400}
        height={300}
        className="w-full border bg-white rounded"
      />
    </div>
  );
};

// Main Application Component
const KGARevionApp = () => {
  const [question, setQuestion] = useState('');
  const [questionType, setQuestionType] = useState('multiple_choice');
  const [candidates, setCandidates] = useState(['', '', '', '']);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [processingStage, setProcessingStage] = useState('');
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [user, setUser] = useState(null);
  
  // WebSocket connection for real-time updates
  const { isConnected, lastMessage, sendMessage } = useWebSocket(`${API_BASE.replace('http', 'ws')}/ws/medical-qa`);
  
  // Load metrics on mount
  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // Update every 30s
    return () => clearInterval(interval);
  }, []);
  
  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'status') {
        setProcessingStage(lastMessage.message);
      } else if (lastMessage.type === 'answer') {
        setResult({
          answer: lastMessage.data.text,
          confidence: lastMessage.data.confidence,
          triplets: []
        });
        setIsProcessing(false);
      } else if (lastMessage.type === 'triplet') {
        setResult(prev => ({
          ...prev,
          triplets: [...(prev?.triplets || []), lastMessage.data]
        }));
      }
    }
  }, [lastMessage]);
  
  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/metrics`);
      const data = await response.json();
      setMetrics(data);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
  };
  
  const handleSubmit = async () => {
    setError(null);
    setResult(null);
    setIsProcessing(true);
    setProcessingStage('Initializing...');
    
    try {
      if (isConnected) {
        // Use WebSocket for real-time updates
        sendMessage({
          text: question,
          question_type: questionType,
          candidates: questionType === 'multiple_choice' ? candidates.filter(c => c) : null
        });
      } else {
        // Fallback to REST API
        const response = await fetch(`${API_BASE}/api/medical-qa`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': user ? `Bearer ${user.token}` : ''
          },
          body: JSON.stringify({
            text: question,
            question_type: questionType,
            candidates: questionType === 'multiple_choice' ? candidates.filter(c => c) : null
          })
        });
        
        if (!response.ok) {
          throw new Error(`Error: ${response.statusText}`);
        }
        
        const data = await response.json();
        setResult({
          answer: data.answer,
          confidence: data.confidence_score,
          triplets: data.verified_triplets,
          entities: data.medical_entities,
          processingTime: data.processing_time_ms
        });
        
        // Add to history
        setHistory(prev => [{
          question,
          answer: data.answer,
          timestamp: new Date().toISOString()
        }, ...prev].slice(0, 10));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
      setProcessingStage('');
    }
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <Brain className="w-8 h-8 text-indigo-600" />
              <h1 className="text-xl font-bold text-gray-900">KGAREVION Medical QA</h1>
              {isConnected && (
                <span className="flex items-center text-green-600 text-sm">
                  <span className="w-2 h-2 bg-green-600 rounded-full mr-1 animate-pulse"></span>
                  Live
                </span>
              )}
            </div>
            
            <div className="flex items-center space-x-4">
              {metrics && (
                <div className="flex items-center space-x-3 text-sm text-gray-600">
                  <div className="flex items-center">
                    <Activity className="w-4 h-4 mr-1" />
                    {metrics.total_questions_processed} queries
                  </div>
                  <div>
                    {metrics.average_processing_time_ms}ms avg
                  </div>
                  <div className="text-green-600">
                    {(metrics.model_accuracy * 100).toFixed(1)}% accuracy
                  </div>
                </div>
              )}
              
              {user ? (
                <div className="flex items-center space-x-2">
                  <User className="w-5 h-5 text-gray-600" />
                  <span className="text-sm text-gray-700">{user.username}</span>
                  <button className="p-1 hover:bg-gray-100 rounded">
                    <LogOut className="w-4 h-4 text-gray-600" />
                  </button>
                </div>
              ) : (
                <button className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 text-sm">
                  Sign In
                </button>
              )}
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Question Input Panel */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <h2 className="text-lg font-semibold">Ask a Medical Question</h2>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Question Type
                    </label>
                    <div className="flex gap-4">
                      <label className="flex items-center">
                        <input
                          type="radio"
                          value="multiple_choice"
                          checked={questionType === 'multiple_choice'}
                          onChange={(e) => setQuestionType(e.target.value)}
                          className="mr-2"
                        />
                        Multiple Choice
                      </label>
                      <label className="flex items-center">
                        <input
                          type="radio"
                          value="open_ended"
                          checked={questionType === 'open_ended'}
                          onChange={(e) => setQuestionType(e.target.value)}
                          className="mr-2"
                        />
                        Open Ended
                      </label>
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Your Question
                    </label>
                    <textarea
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="e.g., Which protein is associated with Retinitis Pigmentosa 59?"
                      className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      rows="3"
                      required
                    />
                  </div>
                  
                  {questionType === 'multiple_choice' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Answer Candidates
                      </label>
                      <div className="space-y-2">
                        {candidates.map((candidate, index) => (
                          <div key={index} className="flex items-center gap-2">
                            <span className="font-medium">{String.fromCharCode(65 + index)}:</span>
                            <input
                              type="text"
                              value={candidate}
                              onChange={(e) => {
                                const newCandidates = [...candidates];
                                newCandidates[index] = e.target.value;
                                setCandidates(newCandidates);
                              }}
                              placeholder={`Option ${String.fromCharCode(65 + index)}`}
                              className="flex-1 px-3 py-1 border rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <button
                    type="button"
                    onClick={handleSubmit}
                    disabled={isProcessing}
                    className="w-full flex items-center justify-center px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        {processingStage || 'Processing...'}
                      </>
                    ) : (
                      <>
                        <Send className="w-4 h-4 mr-2" />
                        Submit Question
                      </>
                    )}
                  </button>
                                  </div>
              </CardContent>
            </Card>
            
            {/* Results Display */}
            {result && (
              <Card className="border-green-200 bg-green-50">
                <CardHeader>
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    Answer
                  </h3>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="p-4 bg-white rounded-lg border border-green-200">
                    <p className="text-lg">{result.answer}</p>
                    {result.confidence && (
                      <div className="mt-2 flex items-center gap-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-green-600 h-2 rounded-full"
                            style={{ width: `${result.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-sm text-gray-600">
                          {(result.confidence * 100).toFixed(1)}% confidence
                        </span>
                      </div>
                    )}
                  </div>
                  
                  {result.entities && result.entities.length > 0 && (
                    <div>
                      <h4 className="text-sm font-semibold mb-2">Medical Entities Identified:</h4>
                      <div className="flex flex-wrap gap-2">
                        {result.entities.map((entity, idx) => (
                          <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                            {entity}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {result.triplets && result.triplets.length > 0 && (
                    <TripletGraph triplets={result.triplets} />
                  )}
                  
                  {result.processingTime && (
                    <p className="text-sm text-gray-600">
                      Processing time: {result.processingTime}ms
                    </p>
                  )}
                </CardContent>
              </Card>
            )}
            
            {error && (
              <Alert className="border-red-200 bg-red-50">
                <XCircle className="w-4 h-4 text-red-600" />
                <AlertDescription className="text-red-800">
                  {error}
                </AlertDescription>
              </Alert>
            )}
          </div>
          
          {/* Side Panel */}
          <div className="space-y-6">
            {/* Question History */}
            <Card>
              <CardHeader 
                className="cursor-pointer"
                onClick={() => setShowHistory(!showHistory)}
              >
                <h3 className="text-lg font-semibold flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <History className="w-5 h-5" />
                    Recent Questions
                  </span>
                  <ChevronDown className={`w-4 h-4 transition-transform ${showHistory ? 'rotate-180' : ''}`} />
                </h3>
              </CardHeader>
              {showHistory && (
                <CardContent>
                  {history.length === 0 ? (
                    <p className="text-sm text-gray-500">No recent questions</p>
                  ) : (
                    <div className="space-y-3 max-h-64 overflow-y-auto">
                      {history.map((item, idx) => (
                        <div key={idx} className="border-b pb-2 last:border-b-0">
                          <p className="text-sm font-medium truncate">{item.question}</p>
                          <p className="text-xs text-gray-600 truncate">Answer: {item.answer}</p>
                          <p className="text-xs text-gray-400">
                            {new Date(item.timestamp).toLocaleTimeString()}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              )}
            </Card>
            
            {/* Processing Pipeline Status */}
            {isProcessing && (
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">Processing Pipeline</h3>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {['Extracting Entities', 'Generating Triplets', 'Reviewing with KG', 'Generating Answer'].map((stage, idx) => (
                      <div key={idx} className="flex items-center gap-3">
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
                          processingStage.toLowerCase().includes(stage.toLowerCase().split(' ')[0]) 
                            ? 'bg-indigo-600 text-white' 
                            : 'bg-gray-200 text-gray-400'
                        }`}>
                          {processingStage.toLowerCase().includes(stage.toLowerCase().split(' ')[0]) 
                            ? <Loader2 className="w-3 h-3 animate-spin" />
                            : idx + 1
                          }
                        </div>
                        <span className="text-sm">{stage}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
            
            {/* System Info */}
            <Card>
              <CardHeader>
                <h3 className="text-sm font-semibold text-gray-600">System Information</h3>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Model:</span>
                    <span className="font-medium">LLaMA3-8B</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Knowledge Graph:</span>
                    <span className="font-medium">PrimeKG</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Triplets:</span>
                    <span className="font-medium">4M+</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Status:</span>
                    <span className="font-medium text-green-600">Online</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
};

export default KGARevionApp;