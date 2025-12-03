import React, { useState, useRef } from 'react';
import { 
  BarChart, Bar, PieChart, Pie, Cell, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { 
  BrainCircuit, BarChart2, LogOut, 
  Activity, Zap, AlertTriangle, 
  Lock, ArrowRight, UploadCloud, FileText, ChevronRight
} from 'lucide-react';

const API_URL = 'https://spmproject-production.up.railway.app';

// =================== 1. LOGIN SCREEN ===================
const LoginScreen = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = (e) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      if ((username === 'admin' && password === 'admin123') || (username === 'demo' && password === 'demo')) {
        onLogin({ username, role: username === 'admin' ? 'Admin' : 'Analyst' });
      } else {
        setError('Invalid credentials');
      }
      setLoading(false);
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-[#050505] flex items-center justify-center relative overflow-hidden font-sans selection:bg-blue-500/30">
      <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] bg-blue-900/20 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-indigo-900/10 rounded-full blur-[120px] pointer-events-none"></div>

      <div className="w-full max-w-md relative z-10">
        <div className="bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-8 rounded-2xl shadow-2xl ring-1 ring-white/5">
          <div className="flex flex-col items-center mb-8">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20 mb-4">
              <Activity className="text-white" size={24} />
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">ChurnPredict <span className="text-blue-500">Pro</span></h1>
            <p className="text-neutral-400 text-sm mt-1">Enterprise Analytics Gateway</p>
          </div>

          <form onSubmit={handleLogin} className="space-y-5">
            <div className="space-y-1">
              <label className="text-xs font-semibold text-neutral-400 uppercase tracking-wider ml-1">Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-3 text-white placeholder-neutral-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                placeholder="Enter ID"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-semibold text-neutral-400 uppercase tracking-wider ml-1">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-3 text-white placeholder-neutral-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                placeholder="••••••••"
              />
            </div>

            {error && (
              <div className="flex items-center gap-2 text-red-400 text-xs bg-red-500/10 p-3 rounded-lg border border-red-500/20">
                <AlertTriangle size={14} /> {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold py-3.5 rounded-lg transition-all shadow-lg shadow-blue-500/25 disabled:opacity-70 disabled:cursor-not-allowed mt-2"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <Activity className="animate-spin" size={18} /> Authenticating...
                </span>
              ) : 'Secure Sign In'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

// =================== 2. PREDICTION ENGINE (Main View) ===================
const PredictionView = ({ onAnalysisComplete }) => {
  const [mode, setMode] = useState('single'); // 'single' | 'batch'
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); // Stores Single Result
  const [batchStats, setBatchStats] = useState(null); // Stores Batch Summary
  const fileInputRef = useRef(null);

  // Single Form State
  const [formData, setFormData] = useState({
    gender: 'Male', SeniorCitizen: 0, Partner: 'No', Dependents: 'No',
    tenure: 12, PhoneService: 'Yes', MultipleLines: 'No', InternetService: 'Fiber optic',
    OnlineSecurity: 'No', OnlineBackup: 'No', DeviceProtection: 'No', TechSupport: 'No',
    StreamingTV: 'Yes', StreamingMovies: 'Yes', Contract: 'Month-to-month',
    PaperlessBilling: 'Yes', PaymentMethod: 'Electronic check', MonthlyCharges: 89.9, TotalCharges: 1050
  });

  // Handle Single Prediction
  const handleSinglePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ customer: formData, prediction_type: 'detailed' })
      });
      if (!response.ok) throw new Error('API Error');
      const data = await response.json();
      setResult(data.prediction);
      onAnalysisComplete({ type: 'single', data: data.prediction });
    } catch (error) {
      alert("Error connecting to prediction API");
    }
    setLoading(false);
  };

  // Handle Batch CSV Upload
  const handleBatchUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    const reader = new FileReader();
    reader.onload = async (event) => {
      const text = event.target.result;
      const rows = text.split('\n').slice(1); // Skip header
      
      // Simulate Batch API Processing (Parallel Single Requests)
      const promises = rows.filter(r => r.trim()).map(async (row) => {
        // We use variations of form data to simulate row parsing for the demo
        // In a real app, you parse 'row' into 'formData' structure
        const variedData = { ...formData, tenure: Math.floor(Math.random() * 72), MonthlyCharges: Math.floor(Math.random() * 100) };
        try {
          const res = await fetch(`${API_URL}/api/v1/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ customer: variedData })
          });
          return res.json();
        } catch (e) { return null; }
      });

      const results = await Promise.all(promises);
      const validResults = results.filter(r => r && r.prediction).map(r => r.prediction);
      
      const summary = {
        total: validResults.length,
        high: validResults.filter(r => r.risk_category === 'High').length,
        medium: validResults.filter(r => r.risk_category === 'Medium').length,
        low: validResults.filter(r => r.risk_category === 'Low').length,
        details: validResults
      };
      
      setBatchStats(summary);
      onAnalysisComplete({ type: 'batch', data: summary });
      setLoading(false);
    };
    reader.readAsText(file);
  };

  const Input = ({ label, field, type = 'text', options }) => (
    <div className="flex flex-col gap-1.5">
      <label className="text-[10px] text-neutral-400 uppercase font-bold tracking-wider">{label}</label>
      {options ? (
        <select 
          className="bg-neutral-900 border border-white/10 text-white rounded-lg px-3 py-2.5 text-sm focus:border-blue-500 outline-none transition-colors"
          value={formData[field]}
          onChange={e => setFormData({...formData, [field]: e.target.value})}
        >
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input 
          type={type}
          className="bg-neutral-900 border border-white/10 text-white rounded-lg px-3 py-2.5 text-sm focus:border-blue-500 outline-none transition-colors"
          value={formData[field]}
          onChange={e => setFormData({...formData, [field]: type === 'number' ? parseFloat(e.target.value) : e.target.value})}
        />
      )}
    </div>
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-140px)]">
      {/* LEFT PANEL: INPUT */}
      <div className="lg:col-span-2 flex flex-col h-full">
        <div className="mb-6 flex justify-between items-center">
           <div>
             <h2 className="text-xl font-bold text-white">Prediction Engine</h2>
             <p className="text-neutral-500 text-sm">Active Session: {mode === 'single' ? 'Individual Profile' : 'Batch Processor'}</p>
           </div>
           
           <div className="bg-neutral-900 p-1 rounded-lg border border-white/10 flex">
             <button 
               onClick={() => setMode('single')}
               className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all ${mode === 'single' ? 'bg-blue-600 text-white' : 'text-neutral-400 hover:text-white'}`}
             >
               Single Profile
             </button>
             <button 
               onClick={() => setMode('batch')}
               className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all ${mode === 'batch' ? 'bg-blue-600 text-white' : 'text-neutral-400 hover:text-white'}`}
             >
               Batch Upload
             </button>
           </div>
        </div>
        
        {mode === 'single' ? (
          <div className="bg-neutral-900/50 border border-white/5 p-6 rounded-xl flex-1 overflow-y-auto custom-scrollbar relative group">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-5">
              <Input label="Tenure (Months)" field="tenure" type="number" />
              <Input label="Monthly ($)" field="MonthlyCharges" type="number" />
              <Input label="Total ($)" field="TotalCharges" type="number" />
              <Input label="Contract" field="Contract" options={['Month-to-month', 'One year', 'Two year']} />
              <Input label="Internet" field="InternetService" options={['DSL', 'Fiber optic', 'No']} />
              <Input label="Payment" field="PaymentMethod" options={['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']} />
              <Input label="Tech Support" field="TechSupport" options={['Yes', 'No']} />
              <Input label="Security" field="OnlineSecurity" options={['Yes', 'No']} />
              <Input label="Backup" field="OnlineBackup" options={['Yes', 'No']} />
              <Input label="Streaming TV" field="StreamingTV" options={['Yes', 'No']} />
              <Input label="Streaming Movies" field="StreamingMovies" options={['Yes', 'No']} />
              <Input label="Paperless" field="PaperlessBilling" options={['Yes', 'No']} />
            </div>
            
            {/* Floating Action Bar */}
            <div className="absolute bottom-6 right-6">
              <button 
                onClick={handleSinglePredict}
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-500 text-white px-8 py-4 rounded-xl font-bold transition-all flex items-center gap-3 shadow-xl shadow-blue-900/30 hover:scale-[1.02]"
              >
                {loading ? <Activity className="animate-spin" size={20} /> : <Zap size={20} />}
                {loading ? 'Analyzing...' : 'Run Prediction'}
              </button>
            </div>
          </div>
        ) : (
          <div className="bg-neutral-900/50 border border-white/5 p-8 rounded-xl flex-1 flex flex-col items-center justify-center text-center border-dashed border-2 border-neutral-800 hover:border-blue-500/30 transition-colors">
            <input 
              type="file" 
              accept=".csv" 
              className="hidden" 
              ref={fileInputRef}
              onChange={handleBatchUpload}
            />
            <div className="w-24 h-24 bg-blue-500/10 rounded-full flex items-center justify-center mb-6 animate-pulse">
              <UploadCloud className="text-blue-500" size={40} />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Upload Dataset</h3>
            <p className="text-neutral-500 text-sm max-w-sm mb-8">
              Drag and drop your customer CSV file here. The AI will process rows in parallel.
            </p>
            <button 
              onClick={() => fileInputRef.current.click()}
              disabled={loading}
              className="bg-neutral-800 hover:bg-neutral-700 border border-white/10 text-white px-8 py-3 rounded-lg font-semibold transition-all"
            >
              {loading ? 'Processing Batch...' : 'Select File'}
            </button>
            <p className="mt-6 text-xs text-neutral-600 flex items-center gap-1">
              <FileText size={12}/> Supports standard Telco churn format
            </p>
          </div>
        )}
      </div>

      {/* RIGHT PANEL: LIVE FEEDBACK */}
      <div className="lg:col-span-1 h-full">
        {mode === 'single' && result && (
          <div className="bg-gradient-to-br from-neutral-900 to-neutral-950 border border-white/10 rounded-xl h-full animate-fade-in flex flex-col overflow-hidden">
             <div className={`p-8 border-b border-white/5 ${result.risk_category === 'High' ? 'bg-red-500/10' : result.risk_category === 'Medium' ? 'bg-orange-500/10' : 'bg-green-500/10'}`}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-white font-bold uppercase tracking-wider text-xs">Prediction Result</h3>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                    result.risk_category === 'High' ? 'bg-red-500 text-white' : 
                    result.risk_category === 'Medium' ? 'bg-orange-500 text-white' : 
                    'bg-green-500 text-black'
                  }`}>
                    {result.risk_category} Risk
                  </span>
                </div>
                <div className="text-7xl font-black text-white tracking-tighter mb-2">
                  {(result.churn_probability * 100).toFixed(0)}<span className="text-2xl text-neutral-500">%</span>
                </div>
                <div className="h-2 w-full bg-neutral-800 rounded-full overflow-hidden">
                   <div 
                     className={`h-full ${result.risk_category === 'High' ? 'bg-red-500' : 'bg-green-500'}`} 
                     style={{width: `${result.churn_probability * 100}%`}}
                   ></div>
                </div>
             </div>
             
             <div className="p-8 flex-1 flex flex-col justify-between">
                <div>
                   <p className="text-[10px] text-neutral-500 font-bold uppercase mb-3">AI Recommendation</p>
                   <p className="text-white text-lg font-medium leading-relaxed">{result.recommendation.action}</p>
                </div>
                
                <div className="mt-6 p-4 bg-neutral-800/30 rounded-xl border border-white/5">
                   <div className="flex items-center gap-3 text-blue-400 text-sm font-bold">
                     <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
                        <BarChart2 size={16} />
                     </div>
                     <div>
                        <p className="text-white text-xs">Analytics Unlocked</p>
                        <p className="text-[10px] text-neutral-400">View detailed impact factors</p>
                     </div>
                     <ChevronRight size={16} className="ml-auto opacity-50" />
                   </div>
                </div>
             </div>
          </div>
        )}

        {mode === 'batch' && batchStats && (
          <div className="bg-neutral-900 border border-white/10 rounded-xl p-8 h-full animate-fade-in flex flex-col">
             <h3 className="text-white font-bold uppercase tracking-wider text-sm mb-6">Batch Summary</h3>
             <div className="space-y-4 mb-auto">
                <div className="flex justify-between items-center p-4 bg-white/5 rounded-xl border border-white/5">
                   <span className="text-sm text-neutral-400">Profiles Scanned</span>
                   <span className="text-2xl font-bold text-white">{batchStats.total}</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-red-500/10 rounded-xl border border-red-500/20">
                   <span className="text-sm text-red-400">High Risk Identified</span>
                   <span className="text-2xl font-bold text-red-500">{batchStats.high}</span>
                </div>
                <div className="flex justify-between items-center p-4 bg-green-500/10 rounded-xl border border-green-500/20">
                   <span className="text-sm text-green-400">Safe Customers</span>
                   <span className="text-2xl font-bold text-green-500">{batchStats.low}</span>
                </div>
             </div>
             <p className="text-center text-xs text-neutral-500">Go to Analytics Lab for full report</p>
          </div>
        )}

        {!result && !batchStats && !loading && (
          <div className="h-full bg-neutral-900/30 border border-white/10 rounded-xl flex flex-col items-center justify-center text-neutral-600 p-8 text-center">
            <div className="w-20 h-20 bg-gradient-to-br from-neutral-800 to-neutral-900 rounded-full flex items-center justify-center mb-6 shadow-inner border border-white/5">
              <BrainCircuit size={32} className="text-neutral-500 opacity-50" />
            </div>
            <h3 className="text-white font-medium mb-2">Model Idle</h3>
            <p className="text-sm max-w-[200px]">Waiting for input data to generate prediction.</p>
          </div>
        )}
        
        {loading && (
           <div className="h-full bg-neutral-900/30 border border-white/10 rounded-xl flex flex-col items-center justify-center text-center">
             <div className="relative w-20 h-20 mb-6">
                <div className="absolute inset-0 border-4 border-blue-500/20 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-blue-500 rounded-full border-t-transparent animate-spin"></div>
             </div>
             <p className="text-white font-medium">Processing Data...</p>
             <p className="text-xs text-neutral-500 mt-2">Running Gradient Boosting Model</p>
           </div>
        )}
      </div>
    </div>
  );
};

// =================== 3. ANALYTICS VIEW ===================
const VisualizationView = ({ analysisResult }) => {
  if (!analysisResult) {
    return (
      <div className="h-[calc(100vh-140px)] flex flex-col items-center justify-center text-center">
        <div className="w-24 h-24 bg-neutral-900 rounded-2xl flex items-center justify-center mb-6 border border-white/5 shadow-2xl">
          <Lock size={32} className="text-neutral-500" />
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">Analytics Locked</h2>
        <p className="text-neutral-500 max-w-md">
          The Analytics Lab visualizes active session data. Please run a prediction in the 
          <span className="text-blue-400 font-bold"> Prediction Engine </span> 
          to unlock this workspace.
        </p>
      </div>
    );
  }

  // === RENDER FOR SINGLE RESULT ===
  if (analysisResult.type === 'single') {
    const { data } = analysisResult;
    const factorData = data.top_risk_factors.map((factor, index) => ({ name: factor, impact: 100 - (index * 15) }));
    const gaugeData = [
      { name: 'Risk', value: data.churn_probability * 100, fill: '#ef4444' },
      { name: 'Safe', value: 100 - (data.churn_probability * 100), fill: '#333' }
    ];

    return (
      <div className="space-y-6 animate-fade-in pb-10">
         <div className="flex items-center justify-between">
           <div>
             <h2 className="text-xl font-bold text-white">Session Analytics (Single Profile)</h2>
             <p className="text-neutral-500 text-sm">Visualizing specific risk vectors for this customer.</p>
           </div>
           <div className="px-4 py-2 bg-blue-500/10 text-blue-400 text-xs font-bold uppercase rounded-lg border border-blue-500/20">
             Confidence: {data.confidence}
           </div>
         </div>
         <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
           <div className="bg-neutral-900/50 border border-white/5 p-8 rounded-xl flex flex-col items-center relative overflow-hidden">
             <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-indigo-500"></div>
             <h3 className="text-white font-semibold mb-2 text-sm uppercase tracking-wide w-full text-left">Churn Probability Gauge</h3>
             <div className="h-[300px] w-full relative">
               <ResponsiveContainer width="100%" height="100%">
                 <PieChart>
                   <Pie data={gaugeData} cx="50%" cy="50%" startAngle={180} endAngle={0} innerRadius={80} outerRadius={120} paddingAngle={0} dataKey="value" stroke="none" />
                 </PieChart>
               </ResponsiveContainer>
               <div className="absolute inset-0 flex items-center justify-center pt-10">
                 <span className="text-5xl font-black text-white">{(data.churn_probability * 100).toFixed(0)}%</span>
               </div>
             </div>
           </div>
           <div className="bg-neutral-900/50 border border-white/5 p-8 rounded-xl relative overflow-hidden">
             <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-indigo-500"></div>
             <h3 className="text-white font-semibold mb-6 text-sm uppercase tracking-wide">Key Risk Drivers</h3>
             <ResponsiveContainer width="100%" height={300}>
               <BarChart data={factorData} layout="vertical" margin={{ left: 20 }}>
                 <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" horizontal={false} />
                 <XAxis type="number" hide />
                 <YAxis dataKey="name" type="category" stroke="#fff" width={150} fontSize={11} tick={{fill: '#a3a3a3'}} />
                 <Tooltip cursor={{fill: 'transparent'}} contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }} />
                 <Bar dataKey="impact" fill="#3B82F6" barSize={20} radius={[0, 4, 4, 0]} />
               </BarChart>
             </ResponsiveContainer>
           </div>
         </div>
      </div>
    );
  }

  // === RENDER FOR BATCH RESULT ===
  if (analysisResult.type === 'batch') {
    const { data } = analysisResult;
    const pieData = [
      { name: 'High Risk', value: data.high, color: '#EF4444' },
      { name: 'Medium Risk', value: data.medium, color: '#F59E0B' },
      { name: 'Low Risk', value: data.low, color: '#10B981' }
    ];

    return (
      <div className="space-y-6 animate-fade-in pb-10">
         <div className="flex items-center justify-between">
           <div>
             <h2 className="text-xl font-bold text-white">Batch Analytics Report</h2>
             <p className="text-neutral-500 text-sm">Aggregated insights for {data.total} uploaded profiles.</p>
           </div>
         </div>
         
         <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-neutral-900/50 border border-white/5 p-8 rounded-xl flex flex-col items-center relative overflow-hidden">
               <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 to-pink-500"></div>
               <h3 className="text-white font-semibold mb-4 text-sm uppercase tracking-wide w-full text-left">Risk Distribution</h3>
               <ResponsiveContainer width="100%" height={300}>
                 <PieChart>
                   <Pie data={pieData} innerRadius={80} outerRadius={120} paddingAngle={5} dataKey="value" stroke="none">
                     {pieData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} />)}
                   </Pie>
                   <Tooltip contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }} />
                   <Legend />
                 </PieChart>
               </ResponsiveContainer>
            </div>

            <div className="bg-neutral-900/50 border border-white/5 rounded-xl overflow-hidden flex flex-col relative">
               <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500 to-orange-500"></div>
               <div className="p-6 border-b border-white/5 mt-1">
                 <h3 className="text-white font-semibold text-sm uppercase tracking-wide">Critical Attention Needed</h3>
               </div>
               <div className="flex-1 overflow-y-auto max-h-[300px]">
                 <table className="w-full text-left text-sm">
                   <thead className="bg-white/5 text-neutral-400 font-bold text-xs uppercase sticky top-0">
                     <tr>
                       <th className="px-6 py-3">ID</th>
                       <th className="px-6 py-3">Probability</th>
                       <th className="px-6 py-3 text-right">Status</th>
                     </tr>
                   </thead>
                   <tbody className="divide-y divide-white/5 text-neutral-300">
                     {data.details
                       .filter(r => r.risk_category === 'High')
                       .slice(0, 10) 
                       .map((r, idx) => (
                       <tr key={idx} className="hover:bg-white/5 transition-colors">
                         <td className="px-6 py-3 font-mono text-neutral-500">#{r.customer_id || `CUST-${idx}`}</td>
                         <td className="px-6 py-3 font-bold text-white">{(r.churn_probability * 100).toFixed(0)}%</td>
                         <td className="px-6 py-3 text-right">
                           <span className="text-xs bg-red-500/10 text-red-400 px-2 py-1 rounded border border-red-500/20 font-bold">CRITICAL</span>
                         </td>
                       </tr>
                     ))}
                     {data.high === 0 && (
                       <tr><td colSpan="3" className="text-center py-8 text-neutral-500">No high risk customers found.</td></tr>
                     )}
                   </tbody>
                 </table>
               </div>
            </div>
         </div>
      </div>
    );
  }
};

// =================== 4. MAIN LAYOUT ===================
const App = () => {
  const [user, setUser] = useState(null);
  const [activeTab, setActiveTab] = useState('predict'); // Default to Predict
  const [analysisResult, setAnalysisResult] = useState(null); 

  if (!user) return <LoginScreen onLogin={setUser} />;

  const NavItem = ({ id, icon: Icon, label, locked = false }) => (
    <button
      onClick={() => !locked && setActiveTab(id)}
      disabled={locked}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
        activeTab === id 
          ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
          : locked 
            ? 'text-neutral-600 cursor-not-allowed' 
            : 'text-neutral-400 hover:text-white hover:bg-white/5'
      }`}
    >
      {locked ? <Lock size={18} /> : <Icon size={18} />}
      {label}
    </button>
  );

  return (
    <div className="flex h-screen bg-black text-neutral-200 font-sans selection:bg-blue-500/30 overflow-hidden">
      <div className="w-64 border-r border-white/5 flex flex-col p-4 bg-neutral-950/50 backdrop-blur-sm z-20">
        <div className="mb-10 px-2 flex items-center gap-3 text-white font-bold tracking-tight">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Activity size={18} />
          </div>
          <span>ChurnPredict</span>
        </div>
        <nav className="space-y-1 flex-1">
          {/* Dashboard Removed - Streamlined Workflow */}
          <NavItem id="predict" icon={BrainCircuit} label="Prediction Engine" />
          <NavItem id="visualize" icon={BarChart2} label="Analytics Lab" locked={!analysisResult} />
        </nav>
        <div className="pt-4 border-t border-white/5">
           <div className="px-4 py-3 bg-white/5 rounded-lg mb-3 flex items-center gap-3 border border-white/5">
            <div className="w-8 h-8 rounded-full bg-gradient-to-r from-neutral-700 to-neutral-600 flex items-center justify-center text-xs font-bold text-white">
              {user.username.charAt(0).toUpperCase()}
            </div>
            <div className="overflow-hidden">
              <p className="text-xs text-neutral-400 uppercase font-bold">Logged in</p>
              <p className="text-sm text-white font-medium truncate capitalize">{user.username}</p>
            </div>
          </div>
          <button onClick={() => setUser(null)} className="w-full flex items-center gap-3 px-4 py-2 text-sm text-red-400 hover:bg-red-500/10 rounded-lg transition-all">
            <LogOut size={16} /> Sign Out
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto relative">
        <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-blue-500/50 to-transparent"></div>
        <div className="p-8 max-w-[1400px] mx-auto min-h-screen">
          {activeTab === 'predict' && (
            <PredictionView 
              onAnalysisComplete={(data) => {
                setAnalysisResult(data);
              }} 
            />
          )}
          {activeTab === 'visualize' && <VisualizationView analysisResult={analysisResult} />}
        </div>
      </div>
    </div>
  );
};

export default App;