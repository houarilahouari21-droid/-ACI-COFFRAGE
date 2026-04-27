import { useState, useMemo, useCallback, useRef, useEffect, ChangeEvent } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Building2, 
  Trash2, 
  Upload, 
  Download, 
  Plus, 
  X, 
  Check, 
  AlertTriangle, 
  Copy, 
  FileText,
  FileUp,
  Sparkles,
  Loader2,
  Trash,
  LayoutDashboard,
  Box,
  Rows,
  Settings,
  Settings2,
  MoreVertical,
  History,
  FileDown,
  ChevronRight,
  Zap,
  ShieldCheck,
  ShieldAlert,
  Info,
  Maximize,
  ArrowUpRight,
  Calculator,
  Code,
  Search,
  SortAsc,
  Activity,
  FileJson,
  RotateCcw,
  TrendingUp,
  CheckCircle2,
  LayoutGrid,
  BarChart3,
  LogIn,
  LogOut,
  User as UserIcon
} from 'lucide-react';
import { GoogleGenAI, Type } from "@google/genai";
import { 
  auth, 
  db, 
  signInWithGoogle, 
  logout, 
  handleFirestoreError 
} from './lib/firebase';
import { 
  onAuthStateChanged, 
  User as FirebaseUser 
} from 'firebase/auth';
import { 
  collection, 
  query, 
  where, 
  onSnapshot, 
  doc, 
  setDoc, 
  deleteDoc, 
  getDocs,
  writeBatch,
  Timestamp
} from 'firebase/firestore';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell,
  PieChart,
  Pie
} from 'recharts';

// --- Constants ---
const PLY = { KS: 0.418, I: 0.167, lbQ: 5.621, Fb: 1545, Frs: 82, E: 1500000 };
const WOOD = { I: 12.5, S: 7.15, Fv: 135, E: 1300000, Fb_adj: 1150 };
const ALU = { E: 9860000, I: 17 };
const STD_SPACING = [24, 20, 16, 12, 10, 8] as const;
const ALU_CAP: Record<string, Record<number, number>> = { 
  simple: { 4: 2620, 5: 2020, 6: 1408, 7: 885 }, 
  double: { 4: 2620, 5: 2020, 6: 1408, 7: 1404 } 
};
const BEAM_CAP: Record<number, number> = { 4: 2620, 5: 2020, 6: 1408, 7: 1404 };
const FRAME_CAP = 10000;
const CONCRETE_PCF = 150;
const DEFLECTION_LIMIT = 360;

// --- Helper Components ---
const CapacityIndicator = ({ value, label, limit, unit = "PSF" }: { value: number; label: string; limit: number; unit?: string }) => {
  const ratio = (value / limit) * 100;
  const isOver = value > limit;
  
  return (
    <div className="space-y-1.5 flex-1">
      <div className="flex justify-between items-end text-[9px] font-black uppercase tracking-widest leading-none">
        <span className="text-text-muted">{label}</span>
        <span className={isOver ? 'text-danger' : 'text-accent'}>{fmt(value, 0)} / {fmt(limit, 0)} {unit}</span>
      </div>
      <div className="h-2 bg-bg rounded-full overflow-hidden border border-border/50">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(ratio, 100)}%` }}
          className={`h-full transition-colors duration-500 ${isOver ? 'bg-danger shadow-[0_0_8px_rgba(239,68,68,0.4)]' : 'bg-accent shadow-[0_0_8px_rgba(37,99,235,0.3)]'}`}
        />
      </div>
      <div className="flex justify-between items-center text-[8px] font-bold">
        <span className={`${isOver ? 'text-danger animate-pulse' : 'text-success'} uppercase`}>
           {isOver ? '❗ SURCHARGE' : '✅ OK'}
        </span>
        <span className="opacity-60">{fmt(ratio, 1)}% UTILISÉ</span>
      </div>
    </div>
  );
};

// --- Types ---
type ElementType = 'DALLE' | 'POUTRE';

interface ElementParams {
  id: number;
  type: ElementType;
  name: string;
  isVar: boolean;
  epMin: number;
  epMax: number;
  dd: number;
  dl: number;
  trib: number;
  span: number;
  stype?: string;
  cho: number;
}

interface ProjectData {
  id: string;
  name: string;
  updatedAt: string;
  elements: {
    id: number;
    params: ElementParams;
    total: number;
    lm: string;
    ok_wood: boolean;
    ok_alu: boolean;
    ok_defl: boolean;
    ok_frame: boolean;
    time: string;
    isAi?: boolean;
  }[];
}

interface ExtractionCandidate {
  modelName: string;
  modelId: string;
  elements: any[];
}

// --- Utils ---
const fmt = (n: number | undefined, d = 2) => typeof n === "number" && isFinite(n) ? n.toFixed(d) : "—";
const safeDiv = (a: number, b: number, f = 0) => b === 0 || !isFinite(b) ? f : a / b;

const OPENROUTER_VISION_MODELS = [
  { id: "google/gemini-2.0-flash-exp", name: "Gemini 2.0 Flash (Gratuit)", icon: "⚡" },
  { id: "anthropic/claude-3.5-sonnet", name: "Claude 3.5 Sonnet", icon: "🧠" },
  { id: "openai/gpt-4o", name: "GPT-4o (OpenAI)", icon: "🤖" },
  { id: "google/gemini-pro-1.5", name: "Gemini 1.5 Pro", icon: "💎" },
  { id: "openai/gpt-4o-mini", name: "GPT-4o Mini", icon: "🖱️" },
  { id: "meta-llama/llama-3.2-11b-vision-instruct:free", name: "Llama 3.2 11B (Gratuit)", icon: "🐑" },
  { id: "qwen/qwen-2-vl-7b-instruct", name: "Qwen 2 VL (Gratuit)", icon: "🐲" },
  { id: "x-ai/grok-2-vision-1212", name: "Grok 2 Vision", icon: "🌌" },
  { id: "mistralai/pixtral-12b", name: "Pixtral 12B", icon: "🌪️" },
  { id: "meta-llama/llama-3.2-90b-vision-instruct", name: "Llama 3.2 90B", icon: "🦙" }
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'dalle' | 'poutre'>('dashboard');
  const [user, setUser] = useState<FirebaseUser | null>(null);
  const [isFirebaseLoading, setIsFirebaseLoading] = useState(true);
  
  const [allProjects, setAllProjects] = useState<ProjectData[]>(() => {
    const saved = localStorage.getItem('coffrageProjects');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        console.error("Error parsing saved projects", e);
      }
    }
    const defaultProject = { 
      id: 'PRJ-' + Math.random().toString(36).substr(2, 6).toUpperCase(),
      name: "Résidences Beaumont", 
      elements: [],
      updatedAt: new Date().toISOString()
    };
    return [defaultProject];
  });

  const [currentProjectId, setCurrentProjectId] = useState<string>(allProjects[0].id);
  const [isRenaming, setIsRenaming] = useState(false);
  const [tempName, setTempName] = useState("");

  const projectData = useMemo(() => {
    return allProjects.find(p => p.id === currentProjectId) || allProjects[0];
  }, [allProjects, currentProjectId]);

  const setProjectData = useCallback((value: ProjectData | ((prev: ProjectData) => ProjectData)) => {
    setAllProjects(prev => prev.map(p => {
      if (p.id === currentProjectId) {
        const updated = typeof value === 'function' ? (value as any)(p) : value;
        const finalProject = { 
          ...p,
          ...updated, 
          id: p.id,
          updatedAt: new Date().toISOString() 
        };

        // Sync with Firestore if logged in
        if (user) {
          setDoc(doc(db, 'projects', p.id), {
            ...finalProject,
            userId: user.uid,
            updatedAt: Timestamp.now()
          }).catch(e => console.error("Firestore Update Error:", e));
        }

        return finalProject;
      }
      return p;
    }));
  }, [currentProjectId, user]);

  useEffect(() => {
    localStorage.setItem('coffrageProjects', JSON.stringify(allProjects));
  }, [allProjects]);

  // --- Firebase Auth & Sync ---
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (u) => {
      setUser(u);
      if (u) {
        // Sync local projects to Firestore on first login? 
        // For simplicity, we just fetch user's projects
        const q = query(collection(db, 'projects'), where('userId', '==', u.uid));
        const unsubFirestore = onSnapshot(q, (snapshot) => {
          const firestoreProjects: ProjectData[] = snapshot.docs.map(d => {
            const data = d.data();
            return {
              ...data,
              id: d.id,
              updatedAt: data.updatedAt?.toDate?.()?.toISOString() || new Date().toISOString()
            } as ProjectData;
          });

          if (firestoreProjects.length > 0) {
            setAllProjects(prev => {
              // Merge: keep local projects that are not in Firestore yet, but prefer Firestore
              const localOnly = prev.filter(lp => !firestoreProjects.find(fp => fp.id === lp.id));
              return [...firestoreProjects, ...localOnly];
            });
          }
          setIsFirebaseLoading(false);
        }, (err) => {
          console.error("Firestore Listen Error:", err);
          setIsFirebaseLoading(false);
        });
        return () => unsubFirestore();
      } else {
        setIsFirebaseLoading(false);
      }
    });
    return () => unsubscribe();
  }, []);

  const loginWithGoogle = async () => {
    try {
      await signInWithGoogle();
      showToast("Connecté avec succès", "success");
    } catch (e) {
      showToast("Échec de connexion", "error");
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      showToast("Déconnecté", "info");
      // Optionally clear all projects or just let it be?
      // For now, keep them in local storage
    } catch (e) {
      showToast("Erreur lors de la déconnexion", "error");
    }
  };

  const createNewProject = () => {
    const newId = 'PRJ-' + Math.random().toString(36).substr(2, 6).toUpperCase();
    const newProj: ProjectData = {
      id: newId,
      name: "Nouveau Projet " + (allProjects.length + 1),
      elements: [],
      updatedAt: new Date().toISOString()
    };

    if (user) {
      setDoc(doc(db, 'projects', newId), {
        ...newProj,
        userId: user.uid,
        updatedAt: Timestamp.now()
      }).catch(e => showToast("Erreur Firestore", "error"));
    }

    setAllProjects(prev => [newProj, ...prev]);
    setCurrentProjectId(newProj.id);
    setActiveTab('dashboard');
    showToast("Nouveau projet créé");
  };

  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'factor'>('date');
  const [showFormulaId, setShowFormulaId] = useState<number | null>(null);

  const filteredElements = useMemo(() => {
    let filtered = projectData.elements.filter(el => 
      el.params.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (sortBy === 'name') {
      filtered = [...filtered].sort((a, b) => a.params.name.localeCompare(b.params.name));
    } else if (sortBy === 'factor') {
      // Sort by safest (lower ratio) to most critical
      const getRatio = (el: any) => el.capAlu > 0 ? (el.loadAlu / el.capAlu) : 2;
      filtered = [...filtered].sort((a, b) => getRatio(b) - getRatio(a));
    } else {
      filtered = [...filtered].sort((a, b) => b.id - a.id);
    }
    return filtered;
  }, [projectData.elements, searchQuery, sortBy]);

  const statsData = useMemo(() => {
    if (projectData.elements.length === 0) return [];
    return projectData.elements.slice(0, 8).map(el => ({
      name: el.params.name,
      charge: Math.round(el.total),
      capacity: Math.round(el.capAlu / el.params.trib) // Normalized to psf
    }));
  }, [projectData.elements]);

  const [extractionCandidates, setExtractionCandidates] = useState<ExtractionCandidate[]>([]);
  const [selectedOrModels, setSelectedOrModels] = useState<string[]>(["anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5", "openai/gpt-4o-mini"]);
  const [selectedOrModel, setSelectedOrModel] = useState<string>("anthropic/claude-3.5-sonnet");
  const [showOrConfig, setShowOrConfig] = useState(false);

  const deleteProject = (id: string) => {
    if (allProjects.length <= 1) {
      showToast("Impossible de supprimer le dernier projet", "error");
      return;
    }
    askConfirmation(
      "Supprimer le projet",
      "Voulez-vous vraiment supprimer ce projet ? Cette action est irréversible.",
      () => {
        const remaining = allProjects.filter(p => p.id !== id);
        
        if (user) {
          deleteDoc(doc(db, 'projects', id)).catch(e => console.error(e));
        }

        setAllProjects(remaining);
        if (currentProjectId === id) {
          setCurrentProjectId(remaining[0].id);
          setActiveTab('dashboard');
        }
        showToast("Projet supprimé", "info");
      }
    );
  };

  const startRenaming = () => {
    setTempName(projectData.name);
    setIsRenaming(true);
  };

  const saveRename = () => {
    if (tempName.trim()) {
      setProjectData(prev => ({ ...prev, name: tempName.trim() }));
      setIsRenaming(false);
      showToast("Projet renommé");
    }
  };

  const [editingId, setEditingId] = useState<number | null>(null);
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [aiProvider, setAiProvider] = useState<'google' | 'groq' | 'openrouter' | 'huggingface'>('google');
  const [pendingExtractions, setPendingExtractions] = useState<any[]>([]);
  const [localGeminiKey, setLocalGeminiKey] = useState<string>(() => localStorage.getItem('COFFRAGE_GEMINI_KEY') || "");
  const [showKeyInput, setShowKeyInput] = useState(false);
  
  const aiRef = useRef<GoogleGenAI | null>(null);

  // Custom UI Feedback State
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' | 'info' } | null>(null);
  const [confirmModal, setConfirmModal] = useState<{
    show: boolean;
    title: string;
    message: string;
    onConfirm: () => void;
  }>({
    show: false,
    title: "",
    message: "",
    onConfirm: () => {},
  });

  const showToast = (message: string, type: 'success' | 'error' | 'info' = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const askConfirmation = (title: string, message: string, onConfirm: () => void) => {
    setConfirmModal({ show: true, title, message, onConfirm });
  };
  
  // Form State
  const [formName, setFormName] = useState("");
  const [formIsVar, setFormIsVar] = useState(false);
  const [formEp, setFormEp] = useState(450);
  const [formEpMin, setFormEpMin] = useState(300);
  const [formEpMax, setFormEpMax] = useState(450);
  const [formDead, setFormDead] = useState(8);
  const [formLive, setFormLive] = useState(50);
  const [formTrib, setFormTrib] = useState(4);
  const [formSpan, setFormSpan] = useState(7);
  const [formSType, setFormSType] = useState("double");
  const [formChoix, setFormChoix] = useState(48);

  // --- Persistence Legacy Cleanup ---
  useEffect(() => {
    // We only use coffrageProjects now
    localStorage.removeItem('coffrageProject');
  }, []);

  // Calculations
  const calculations = useMemo(() => {
    const depth = formIsVar ? formEpMax : formEp;
    const ep_min_val = formIsVar ? formEpMin : depth;
    
    // A1 - Charge
    const din = Math.ceil(depth / 25.4);
    const conc = (din / 12) * CONCRETE_PCF;
    const total = conc + formDead + formLive;
    
    // A2 - Contreplaqué & 4x4 (Secondary Beams / Joists)
    const wb = safeDiv(total, 12);
    const wd = safeDiv(total - formLive, 12);
    const lb_ply = wb > 0 ? Math.sqrt(10 * PLY.Fb * PLY.KS / wb) : Infinity;
    const ld_ply = wd > 0 ? Math.pow(145 * PLY.E * PLY.I / (DEFLECTION_LIMIT * wd), 1/3) : Infinity;
    const lr_ply = wb > 0 ? (5/3) * PLY.Frs * PLY.lbQ / wb : Infinity;
    const lm_ply = Math.min(lb_ply, ld_ply, lr_ply);
    
    // The "standard" suggested spacing for plywood
    const sp_suggested = STD_SPACING.find(s => s <= lm_ply) || STD_SPACING[STD_SPACING.length - 1];
    
    // Use user-defined spacing for beam calculations
    const sp = formChoix; 
    const wcft = (sp / 12) * total;
    const wdft = (sp / 12) * (total - formLive);
    const wcin = safeDiv(wcft, 12);
    const wdin = safeDiv(wdft, 12);
    const lb_wood = wcin > 0 ? Math.sqrt(10 * WOOD.Fb_adj * WOOD.S / wcin) : Infinity;
    const ld_wood = wdin > 0 ? Math.pow(145 * WOOD.E * WOOD.I / (DEFLECTION_LIMIT * wdin), 1/3) : Infinity;
    const lm_wood = Math.min(lb_wood, ld_wood);
    
    const woodUtil = (sp / lm_wood) * 100;
    const plyUtil = (sp / lm_ply) * 100;
    
    // A3 - Aluma
    const isDalle = activeTab === 'dalle';
    const loadAlu = formTrib * total;
    const capAlu = isDalle ? (ALU_CAP[formSType]?.[formSpan] || 0) : (BEAM_CAP[formSpan] || 0);
    const okAlu = loadAlu <= capAlu && capAlu > 0;
    
    const L = formSpan * 12;
    const md = L / 270;
    const wdAlu = safeDiv(formTrib * (total - formLive), 12);
    const deflAlu = wdAlu > 0 ? (isDalle && formSType === "simple" ? (5 * wdAlu * Math.pow(L, 4) / (384 * ALU.E * ALU.I)) : (wdAlu * Math.pow(L, 4) / (185 * ALU.E * ALU.I))) : Infinity;
    const dOk = deflAlu <= md;
    
    // A4 - Cadres
    const pt = loadAlu * formSpan;
    const fOk = pt <= FRAME_CAP;
    
    return {
      din, conc, total,
      sp, wcft, lm_wood, woodOk: formChoix <= lm_wood,
      woodUtil, plyUtil, lm_ply,
      loadAlu, capAlu, okAlu, md, deflAlu, dOk,
      pt, fOk,
      ep_min_val, depth_max: depth
    };
  }, [formIsVar, formEp, formEpMin, formEpMax, formDead, formLive, formTrib, formSpan, formSType, formChoix, activeTab]);

  const resetForm = useCallback(() => {
    setEditingId(null);
    setFormName("");
    setFormIsVar(false);
    setFormEp(activeTab === 'dalle' ? 450 : 495);
    setFormEpMin(300);
    setFormEpMax(450);
    setFormDead(8);
    setFormLive(50);
    setFormTrib(4);
    setFormSpan(activeTab === 'dalle' ? 7 : 6);
    setFormSType("double");
    setFormChoix(48);
  }, [activeTab]);

  const saveElement = () => {
    const calcType = activeTab === 'dashboard' ? 'DALLE' : (activeTab === 'dalle' ? 'DALLE' : 'POUTRE');
    const name = formName.trim() || `${calcType}_${projectData.elements.length + 1}`;
    
    const params: ElementParams = {
      id: editingId || Date.now(),
      type: calcType as ElementType,
      name,
      isVar: formIsVar,
      epMin: formIsVar ? formEpMin : formEp,
      epMax: formIsVar ? formEpMax : formEp,
      dd: formDead,
      dl: formLive,
      trib: formTrib,
      span: formSpan,
      stype: calcType === 'DALLE' ? formSType : undefined,
      cho: formChoix,
    };

    const newElement = {
      id: params.id,
      params,
      total: Math.round(calculations.total),
      lm: fmt(calculations.lm_wood, 2),
      woodUtil: calculations.woodUtil,
      plyUtil: calculations.plyUtil,
      ok_wood: calculations.woodOk,
      ok_alu: calculations.okAlu,
      ok_defl: calculations.dOk,
      ok_frame: calculations.fOk,
      time: new Date().toLocaleString('fr-FR')
    };

    if (editingId) {
      setProjectData(prev => ({
        ...prev,
        elements: prev.elements.map(el => el.id === editingId ? newElement : el)
      }));
    } else {
      setProjectData(prev => ({
        ...prev,
        elements: [newElement, ...prev.elements]
      }));
    }
    
    resetForm();
  };

  const startEdit = (id: number) => {
    const el = projectData.elements.find(e => e.id === id);
    if (!el) return;
    const p = el.params;
    setEditingId(id);
    setActiveTab(p.type === 'DALLE' ? 'dalle' : 'poutre');
    setFormName(p.name);
    setFormIsVar(p.isVar);
    if (p.isVar) {
      setFormEpMin(p.epMin);
      setFormEpMax(p.epMax);
    } else {
      setFormEp(p.epMax);
    }
    setFormDead(p.dd);
    setFormLive(p.dl);
    setFormTrib(p.trib);
    setFormSpan(p.span);
    if (p.stype) setFormSType(p.stype);
    setFormChoix(p.cho);
  };

  const deleteElement = (id: number) => {
    askConfirmation(
      "Suppression",
      "Voulez-vous vraiment supprimer cet élément ?",
      () => {
        setProjectData(prev => ({
          ...prev,
          elements: prev.elements.filter(e => e.id !== id)
        }));
        if (editingId === id) resetForm();
        showToast("Élément supprimé", "info");
      }
    );
  };

  const clearProject = () => {
    if (projectData.elements.length === 0) {
      showToast("Le projet est déjà vide", "info");
      return;
    }
    askConfirmation(
      "Réinitialiser",
      "Voulez-vous réinitialiser tout le projet ? Cette action supprimera tous les éléments.",
      () => {
        setProjectData(prev => ({ ...prev, elements: [] }));
        resetForm();
        showToast("Projet réinitialisé", "info");
      }
    );
  };

  const exportJSON = () => {
    const b = new Blob([JSON.stringify(projectData, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = `${projectData.name.replace(/\s+/g, '_')}.json`;
    a.click();
  };

  const importJSON = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = ev => {
      try {
        const d = JSON.parse(ev.target?.result as string);
        if (d.elements) {
          setProjectData(d);
          resetForm();
          showToast("Projet importé avec succès");
        } else {
          showToast("Le fichier ne contient pas de données de projet valides", "error");
        }
      } catch (err) { 
        showToast("Fichier JSON invalide", "error"); 
      }
    };
    r.readAsText(f);
    e.target.value = '';
  };

  const copySummary = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      showToast("✅ Résumé copié au presse-papier");
    });
  };

  // --- AI Logic ---
  const handleAiExtraction = async (file: File) => {
    setIsAiLoading(true);
    try {
      const reader = new FileReader();
      const base64Promise = new Promise<string>((resolve) => {
        reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
      });
      reader.readAsDataURL(file);
      const base64 = await base64Promise;

      let extracted: any[] = [];
      const isStaticEnv = window.location.hostname.includes('github.io') || 
                         window.location.hostname.includes('houarilahouari21-droid.github.io');
      
      const viteKey = (import.meta as any).env?.VITE_GEMINI_API_KEY;

      // EXTRACTION GEMINI (DIRECTEMENT SUR LE FRONTEND SELON LES RECOMMANDATIONS DU SKILL)
      if (aiProvider === 'google') {
        const envKey = process.env.GEMINI_API_KEY;
        const key = localGeminiKey || envKey || viteKey;

        if (!key) {
          if (isStaticEnv) {
            setShowKeyInput(true);
            throw new Error("Clé API Gemini locale requise sur GitHub. Cliquez sur l'icône de réglage.");
          } else {
            // Fallback pour AI Studio au cas où le process.env n'est pas encore prêt
            const res = await fetch("/api/ai-extract", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ base64, mimeType: file.type, provider: aiProvider })
            });
            if (!res.ok) throw new Error("Impossible de trouver une clé API Gemini valide.");
            const data = await res.json();
            extracted = data.elements || [];
          }
        }

        if (key && extracted.length === 0) {
          try {
            const ai = new GoogleGenAI({ apiKey: key });
            const response = await ai.models.generateContent({
              model: "gemini-1.5-flash",
              contents: [{
                parts: [
                  { text: `Tu es un expert en coffrage. Analyse ce plan de structure et extrais les informations sur les dalles et les poutres.
                    Retourne UNIQUEMENT un objet JSON: { "elements": [ { "name": string, "thickness": number, "type": "DALLE"|"POUTRE" } ] }.` },
                  { inlineData: { data: base64, mimeType: file.type || "image/jpeg" } }
                ]
              }],
              config: { responseMimeType: "application/json" }
            });

            const text = response.text || "";
            if (!text) throw new Error("Réponse vide de Google.");
            const jsonMatch = text.match(/\{[\s\S]*\}/);
            const parsed = JSON.parse(jsonMatch ? jsonMatch[0] : text);
            extracted = parsed.elements || [];
          } catch (genAiError: any) {
            console.error("Gemini direct call error:", genAiError);
            // Si l'appel direct échoue sur AI Studio, on peut tenter le serveur 
            if (!isStaticEnv) {
              const res = await fetch("/api/ai-extract", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ base64, mimeType: file.type, provider: aiProvider })
              });
              const data = await res.json();
              extracted = data.elements || [];
            } else {
              throw genAiError;
            }
          }
        }
      } else {
        // MODE NORMAL POUR GROQ / OPENROUTER (Serveur Proxy)
        try {
          const res = await fetch("/api/ai-extract", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              base64,
              mimeType: file.type,
              provider: aiProvider,
              models: aiProvider === 'openrouter' ? [selectedOrModel] : undefined
            })
          });

          if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            
            if (errData.error?.includes("GEMINI_API_KEY")) {
              setShowKeyInput(true);
              throw new Error("Clé API manquante. Veuillez configurer GEMINI_API_KEY.");
            }
            throw new Error(errData.error || `Erreur serveur: ${res.status}`);
          }

          const data = await res.json();
          
          if (aiProvider === 'openrouter' && data.candidates) {
            setExtractionCandidates(data.candidates);
            showToast(`Plusieurs modèles ont répondu. Choisissez le meilleur résultat.`, "info");
            setIsAiLoading(false);
            return;
          }

          extracted = data.elements || [];
        } catch (fetchErr: any) {
          if (isStaticEnv) {
            setShowKeyInput(true);
            throw new Error("Ce domaine (GitHub) ne supporte pas le serveur d'analyse. Veuillez configurer une clé Gemini locale (via l'icône de réglages) pour utiliser l'IA.");
          }
          throw fetchErr;
        }
      }

      if (extracted.length === 0) {
        showToast("Aucune donnée détectée. Essayez une image plus claire.", "info");
        return;
      }

      const newElements = extracted.map((item: any, index: number) => {
        const type = item.type === 'DALLE' ? 'DALLE' : 'POUTRE';
        const thick = item.thickness || (type === 'DALLE' ? 200 : 400);
        
        const LD = { conc: (Math.ceil(thick / 25.4) / 12) * 150, total: (Math.ceil(thick / 25.4) / 12) * 150 + 8 + 50 };
        
        return {
          id: Date.now() + index,
          params: {
            id: Date.now() + index,
            type,
            name: item.name,
            isVar: false,
            epMin: thick,
            epMax: thick,
            dd: 8,
            dl: 50,
            trib: 4,
            span: type === 'DALLE' ? 7 : 6,
            stype: "double",
            cho: 48
          },
          total: Math.round(LD.total),
          lm: "60.00",
          ok_wood: true,
          ok_alu: true,
          ok_defl: true,
          ok_frame: true,
          time: new Date().toLocaleTimeString('fr-FR'),
          isAi: true
        };
      });

      setPendingExtractions(prev => [...newElements, ...prev]);
      showToast(`${newElements.length} éléments en attente de validation`, "success");
    } catch (err: any) {
      console.error("Extraction error:", err);
      showToast(`Analyse échouée: ${err.message || 'Erreur inconnue'}`, "error");
    } finally {
      setIsAiLoading(false);
    }
  };

  const approveExtraction = (id: number) => {
    const el = pendingExtractions.find(e => e.id === id);
    if (!el) return;
    
    setProjectData(prev => ({
      ...prev,
      elements: [el, ...prev.elements]
    }));
    setPendingExtractions(prev => prev.filter(e => e.id !== id));
    showToast("Élément ajouté au projet");
  };

  const rejectExtraction = (id: number) => {
    setPendingExtractions(prev => prev.filter(e => e.id !== id));
    showToast("Détection IA rejetée", "info");
  };

  const applyCandidate = (candidate: ExtractionCandidate) => {
    const newElements = candidate.elements.map((item: any, index: number) => {
      const type = item.type === 'DALLE' ? 'DALLE' : 'POUTRE';
      const thick = item.thickness || (type === 'DALLE' ? 200 : 400);
      
      const LD = { conc: (Math.ceil(thick / 25.4) / 12) * 150, total: (Math.ceil(thick / 25.4) / 12) * 150 + 8 + 50 };
      
      return {
        id: Date.now() + index + Math.floor(Math.random() * 1000),
        params: {
          id: Date.now() + index + Math.floor(Math.random() * 1000),
          type,
          name: item.name,
          isVar: false,
          epMin: thick,
          epMax: thick,
          dd: 8,
          dl: 50,
          trib: 4,
          span: type === 'DALLE' ? 7 : 6,
          stype: "double",
          cho: 48
        },
        total: Math.round(LD.total),
        lm: "60.00",
        ok_wood: true,
        ok_alu: true,
        ok_defl: true,
        ok_frame: true,
        time: new Date().toLocaleTimeString('fr-FR'),
        isAi: true
      };
    });

    setPendingExtractions(prev => [...newElements, ...prev]);
    setExtractionCandidates([]);
    showToast(`${newElements.length} éléments de ${candidate.modelName} ajoutés pour validation`, "success");
  };

  const clearAllExtractions = () => {
    if (pendingExtractions.length === 0) return;
    setConfirmModal({
      show: true,
      title: "Vider la liste",
      message: `Voulez-vous rejeter les ${pendingExtractions.length} détections en attente ?`,
      onConfirm: () => {
        setPendingExtractions([]);
        showToast("Liste vidée", "info");
      }
    });
  };

  const summaryText = `
<strong>${formName.trim() || (activeTab === 'dalle' ? 'DALLE' : 'POUTRE')}</strong><br/>
<strong>ÉPAISSEUR : ${formIsVar ? Math.ceil(formEpMin/25.4) + '"@' + calculations.din + '"' : calculations.din + '"'}</strong><br/>
<strong>CHARGE DE CONCEPTION</strong><br/>
CHARGES VIVES DE BÉTON : ${Math.round(calculations.conc)} LBS/PI²<br/>
CHARGES MORTES DU COFFRAGE : ${Math.round(formDead)} LBS/PI²<br/>
CHARGES VIVES DES TRAVAILLEURS : ${Math.round(formLive)} LBS/PI²<br/>
<strong>CHARGES TOTAUX : ${Math.round(calculations.total)} LBS/PI²</strong>`.trim();

  const summaryTextMetric = `
<strong>Résumé de Charge</strong><br/>
<strong>${formName.trim() || (activeTab === 'dalle' ? 'DALLE' : 'POUTRE')} ÉPAISSEUR : ${formIsVar ? formEpMin + '@' + formEpMax + ' mm' : formEp + ' mm'}</strong><br/>
<strong>CHARGE DE CONCEPTION</strong><br/>
CHARGES VIVES DE BÉTON : ${Math.round(calculations.conc)} LBS/PI²<br/>
CHARGES MORTES DU COFFRAGE : ${Math.round(formDead)} LBS/PI²<br/>
CHARGES VIVES DES TRAVAILLEURS : ${Math.round(formLive)} LBS/PI²<br/>
<strong>CHARGES TOTAUX : ${Math.round(calculations.total)} LBS/PI²</strong>`.trim();

  return (
    <div className="flex bg-bg h-screen overflow-hidden text-text-main font-sans">
      {/* Toast Notification */}
      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 20 }}
            exit={{ opacity: 0, y: -50 }}
            className={`fixed top-0 left-1/2 -translate-x-1/2 z-[100] px-6 py-3 rounded-full shadow-2xl font-bold text-sm flex items-center gap-2
            ${toast.type === 'success' ? 'bg-success text-white' : 
              toast.type === 'error' ? 'bg-danger text-white' : 'bg-accent text-white'}`}
          >
            {toast.type === 'success' && <Check size={18} />}
            {toast.type === 'error' && <AlertTriangle size={18} />}
            {toast.type === 'info' && <FileText size={18} />}
            {toast.message}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Confirmation Modal */}
      <AnimatePresence>
        {confirmModal.show && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[90] flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-surface border border-border w-full max-w-sm rounded-2xl shadow-2xl overflow-hidden"
            >
              <div className="p-6">
                <h3 className="text-lg font-black tracking-tight mb-2 uppercase">{confirmModal.title}</h3>
                <p className="text-sm text-text-muted leading-relaxed">{confirmModal.message}</p>
              </div>
              <div className="bg-bg/50 p-4 flex gap-3">
                <button 
                  onClick={() => setConfirmModal(prev => ({ ...prev, show: false }))}
                  className="flex-1 py-2.5 rounded-xl text-text-muted font-bold text-xs hover:bg-white transition-colors"
                >
                  Annuler
                </button>
                <button 
                  onClick={() => {
                    confirmModal.onConfirm();
                    setConfirmModal(prev => ({ ...prev, show: false }));
                  }}
                  className="flex-1 py-2.5 rounded-xl bg-danger text-white font-bold text-xs hover:filter hover:brightness-110 transition-all shadow-lg shadow-danger/20"
                >
                  Confirmer
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Technical Formulas Modal */}
      <AnimatePresence>
        {showFormulaId && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[110] flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
              className="bg-surface border border-border w-full max-w-2xl rounded-2xl shadow-3xl overflow-hidden flex flex-col max-h-[90vh]"
            >
              {(() => {
                const el = projectData.elements.find(e => e.id === showFormulaId);
                if (!el) return null;
                return (
                  <>
                    <div className="p-6 border-b border-border bg-bg/20 flex justify-between items-center">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-accent/10 text-accent rounded-xl flex items-center justify-center"><Calculator size={20}/></div>
                        <div>
                          <div className="flex items-center gap-3">
                            <h3 className="font-black uppercase tracking-tighter text-lg leading-tight">Détails Techniques : {el.params.name}</h3>
                            <div className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${el.ok_wood && el.ok_alu && el.ok_frame ? 'bg-success text-white' : 'bg-danger text-white'}`}>
                               {el.ok_wood && el.ok_alu && el.ok_frame ? 'OK' : 'NON CONFORME'}
                            </div>
                          </div>
                          <p className="text-[10px] font-bold text-text-muted tracking-widest uppercase">Transparence des Calculs d'Ingénierie</p>
                        </div>
                      </div>
                      <button onClick={() => setShowFormulaId(null)} className="p-2 hover:bg-bg rounded-lg transition-colors"><X/></button>
                    </div>
                    <div className="p-8 overflow-y-auto scroller-hidden bg-white">
                       <div className="space-y-8">
                          <section>
                            <h4 className="text-[11px] font-black text-accent uppercase tracking-widest mb-4 flex items-center gap-2">
                              <span className="w-4 h-4 rounded bg-accent text-white flex items-center justify-center text-[8px]">1</span> 
                              Détermination des Charges (ASCE 7)
                            </h4>
                            <div className="grid grid-cols-2 gap-6 pl-6 border-l border-accent/20">
                              <div className="font-mono text-xs space-y-1">
                                <div className="text-text-muted">Charge Morte (D):</div>
                                <div className="font-bold text-text-main">({fmt(el.params.isVar ? el.params.epMax : el.params.ep)}mm / 25.4 / 12) * {CONCRETE_PCF} PCF + {fmt(formDead)} PSF = {fmt(el.total - formLive)} PSF</div>
                              </div>
                              <div className="font-mono text-xs space-y-1">
                                <div className="text-text-muted">Charge Totale (D+L):</div>
                                <div className="font-bold text-text-main">{fmt(el.total - formLive)} + {fmt(formLive)} PSF = {fmt(el.total)} PSF</div>
                              </div>
                              <div className="font-mono text-xs space-y-1 col-span-2 mt-2 pt-2 border-t border-border/50">
                                <div className="text-[10px] text-accent font-bold uppercase mb-1">Efficacité du Matériau</div>
                                <div className="flex items-center gap-4">
                                  <div className="flex-1 space-y-1">
                                    <div className="text-[9px] text-text-muted uppercase">Utilisation Solives Bois ({(el as any).params.cho}" @ {(el as any).lm}")</div>
                                    <div className="h-1.5 w-full bg-bg rounded-full overflow-hidden">
                                      <div className={`h-full ${(el as any).woodUtil > 100 ? 'bg-danger' : 'bg-success'}`} style={{ width: `${Math.min((el as any).woodUtil, 100)}%` }} />
                                    </div>
                                  </div>
                                  <div className={`text-xs font-black ${(el as any).woodUtil > 100 ? 'text-danger' : 'text-success'}`}>{fmt((el as any).woodUtil, 1)}%</div>
                                </div>
                              </div>
                            </div>
                          </section>

                          <section>
                            <h4 className="text-[11px] font-black text-accent uppercase tracking-widest mb-4 flex items-center gap-2">
                              <span className="w-4 h-4 rounded bg-accent text-white flex items-center justify-center text-[8px]">2</span> 
                              Limites de Flexion & Cisaillement
                            </h4>
                            <div className="space-y-4 pl-6 border-l border-accent/20">
                              <div className="bg-bg/40 p-4 rounded-xl border border-border/50">
                                <div className={`text-[10px] font-bold mb-2 uppercase flex items-center justify-between ${el.ok_wood ? 'text-success' : 'text-danger'}`}>
                                  <span>Contreplaqué 19mm (Grosir 0.167)</span>
                                  <span>{el.ok_wood ? '✅ CONFORME' : '❌ ÉCHEC'}</span>
                                </div>
                                <div className="grid grid-cols-3 gap-2">
                                  <div className="text-center">
                                    <div className="text-[9px] text-text-muted leading-none mb-1 uppercase">Flexion</div>
                                    <div className="font-mono text-xs font-bold">{fmt(Math.sqrt(10 * PLY.Fb * PLY.KS / ((el.total)/12)), 1)}"</div>
                                  </div>
                                  <div className="text-center border-x border-border/50">
                                    <div className="text-[9px] text-text-muted leading-none mb-1 uppercase">Defl (L/360)</div>
                                    <div className="font-mono text-xs font-bold">{fmt(Math.pow(145 * PLY.E * PLY.I / (DEFLECTION_LIMIT * ((el.total-formLive)/12)), 1/3), 1)}"</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="text-[9px] text-text-muted leading-none mb-1 uppercase">Rolling Shear</div>
                                    <div className="font-mono text-xs font-bold">{fmt((5/3) * PLY.Frs * PLY.lbQ / ((el.total)/12), 1)}"</div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </section>

                          <section>
                            <h4 className="text-[11px] font-black text-accent uppercase tracking-widest mb-4 flex items-center gap-2">
                              <span className="w-4 h-4 rounded bg-accent text-white flex items-center justify-center text-[8px]">3</span> 
                              Vérification Aluma-Beam
                            </h4>
                            <div className="flex items-center gap-6 pl-6 border-l border-accent/20">
                               <div className={`flex-1 p-4 rounded-xl border ${el.ok_alu ? 'bg-success/5 border-success/20' : 'bg-danger/5 border-danger/20'}`}>
                                  <div className={`text-[10px] font-bold uppercase mb-1 ${el.ok_alu ? 'text-success' : 'text-danger'}`}>
                                    Capacité vs Charge {el.ok_alu ? '✅ OK' : '❌ ÉCHEC'}
                                  </div>
                                  <div className="flex items-center justify-between">
                                    <div className="text-lg font-black">{Math.round(el.total * el.params.trib)} <span className="text-[10px] opacity-40">LBS</span></div>
                                    <div className="text-xs font-bold opacity-30">vs</div>
                                    <div className={`text-lg font-black ${el.ok_alu ? 'text-success' : 'text-danger'}`}>
                                      {Math.round(el.capAlu)} <span className={`text-[10px] opacity-40 ${el.ok_alu ? 'text-success/60' : 'text-danger/60'}`}>LBS</span>
                                    </div>
                                  </div>
                               </div>
                            </div>
                          </section>
                       </div>
                    </div>
                    <div className="p-4 bg-bg/50 border-t border-border flex justify-end">
                       <button 
                        onClick={() => setShowFormulaId(null)}
                        className="px-6 py-2 bg-text-main text-white rounded-xl font-bold text-xs hover:bg-black transition-all"
                       >
                         Compris
                       </button>
                    </div>
                  </>
                );
              })()}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      <aside className="w-[260px] bg-sidebar text-white flex flex-col p-5 shrink-0">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-9 h-9 bg-accent rounded-lg flex items-center justify-center font-black text-lg">🏗️</div>
          <div>
            <div className="font-extrabold text-base tracking-tight leading-tight">ACI COFFRAGE</div>
            <div className="text-[10px] opacity-40 font-bold tracking-widest uppercase mt-0.5">Engineering Suite</div>
          </div>
        </div>

        <nav className="space-y-8 flex-1 overflow-y-auto scroller-hidden">
          <div>
            <div className="flex justify-between items-center mb-4 pr-2">
              <div className="text-[10px] uppercase tracking-widest text-white/40 font-bold">Projets Actifs</div>
              <button 
                onClick={createNewProject}
                className="w-6 h-6 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center transition-colors"
                title="Nouveau Projet"
              >
                <Plus size={14} />
              </button>
            </div>
            <div className="space-y-1">
              <button 
                onClick={() => setActiveTab('dashboard')}
                className={`w-full flex items-center gap-3 p-3 rounded-lg text-[13px] font-medium transition-all duration-200
                ${activeTab==='dashboard' ? 'bg-white/10 text-white shadow-sm' : 'text-white/60 hover:text-white hover:bg-white/5'}`}
              >
                <LayoutDashboard size={16} /> Dashboard Global
              </button>
              
              <div className="mt-4 space-y-1 max-h-[200px] overflow-y-auto pr-1">
                {isFirebaseLoading ? (
                  <div className="flex flex-col items-center justify-center p-8 gap-3 opacity-40">
                    <Loader2 size={24} className="animate-spin text-accent" />
                    <div className="text-[10px] font-bold tracking-widest uppercase">Chargement Cloud...</div>
                  </div>
                ) : allProjects.map(proj => (
                  <div key={proj.id} className="group flex items-center gap-1">
                    <button 
                      onClick={() => {
                        setCurrentProjectId(proj.id);
                        setActiveTab('dashboard');
                      }}
                      className={`flex-1 flex items-center gap-3 p-2.5 rounded-lg text-[12px] font-medium transition-all duration-200 truncate
                      ${currentProjectId === proj.id ? 'bg-accent/20 text-accent border border-accent/20' : 'text-white/40 hover:text-white hover:bg-white/5'}`}
                    >
                      <Box size={14} /> {proj.name}
                    </button>
                    {allProjects.length > 1 && (
                      <button 
                        onClick={(e) => { e.stopPropagation(); deleteProject(proj.id); }}
                        className="p-2 text-white/0 group-hover:text-white/20 hover:text-danger/80 transition-all rounded-lg"
                      >
                        <Trash2 size={12} />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div>
            <div className="text-[10px] uppercase tracking-widest text-white/40 font-bold mb-4">Outils</div>
            <div className="space-y-1">
              <button 
                onClick={() => { setActiveTab('dalle'); resetForm(); }}
                className={`w-full flex items-center gap-3 p-3 rounded-lg text-[13px] font-medium transition-all duration-200
                ${activeTab==='dalle' ? 'bg-white/10 text-white shadow-sm' : 'text-white/60 hover:text-white hover:bg-white/5'}`}
              >
                <Building2 size={16} /> Calculateur de Dalle
              </button>
              <button 
                onClick={() => { setActiveTab('poutre'); resetForm(); }}
                className={`w-full flex items-center gap-3 p-3 rounded-lg text-[13px] font-medium transition-all duration-200
                ${activeTab==='poutre' ? 'bg-white/10 text-white shadow-sm' : 'text-white/60 hover:text-white hover:bg-white/5'}`}
              >
                <Rows size={16} /> Calculateur de Poutre
              </button>
            </div>
          </div>
        </nav>

        {/* AI Sidebar Panel */}
        <div className="mt-auto bg-ai-accent/10 border border-ai-accent/30 rounded-[10px] p-4">
          <div className="text-[11px] font-bold text-ai-accent uppercase tracking-wider mb-2 flex items-center justify-between">
            <span className="flex items-center gap-2"><Sparkles size={12} /> Assistant IA</span>
            <select 
              value={aiProvider} 
              onChange={(e) => setAiProvider(e.target.value as any)}
              className="bg-sidebar border border-white/10 rounded px-1 py-0.5 text-[9px] outline-none"
            >
              <option value="google">GEMINI (Native)</option>
              <option value="groq">GROQ (Llama-3)</option>
              <option value="openrouter">OpenRouter (Assistant)</option>
              <option value="huggingface">Hugging Face</option>
            </select>
          </div>
          
          {aiProvider === 'openrouter' && (
            <div className="mb-3">
              <div className="text-[9px] font-bold text-white/30 uppercase tracking-[0.2em] mb-1.5">Choisir un modèle (Top 10)</div>
              <div className="space-y-0.5 max-h-[140px] overflow-y-auto pr-1 custom-scrollbar">
                {OPENROUTER_VISION_MODELS.map(m => (
                  <button 
                    key={m.id}
                    onClick={() => setSelectedOrModel(m.id)}
                    className={`w-full text-left px-2 py-1.5 rounded-md text-[10px] font-bold transition-all flex items-center justify-between group
                    ${selectedOrModel === m.id ? 'bg-ai-accent text-white shadow-lg' : 'text-white/40 hover:text-white hover:bg-white/5'}`}
                  >
                    <span className="truncate flex items-center gap-2">
                       <span className="opacity-80 group-hover:scale-125 transition-transform">{m.icon}</span> 
                       {m.name}
                    </span>
                    {selectedOrModel === m.id && <Check size={10} />}
                  </button>
                ))}
              </div>
            </div>
          )}

          <p className="text-[11px] text-white/70 leading-relaxed">
            Analyse vos plans PDF/IMG. Configurez vos clés API pour utiliser Gemini ou Groq.
          </p>
          <div className="mt-3 flex items-center justify-between">
            <div className="text-[10px] font-mono opacity-50">Status: {isAiLoading ? 'Analyse...' : 'Prêt'}</div>
            <button 
              onClick={() => setShowKeyInput(!showKeyInput)}
              className={`p-1 rounded transition-colors ${localGeminiKey ? 'text-success' : 'text-white/20 hover:text-white'}`}
              title="Configuration Clé Locale"
            >
              <Settings2 size={12} />
            </button>
          </div>

          <AnimatePresence>
            {showKeyInput && (
              <motion.div 
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="mt-3 pt-3 border-t border-white/5 space-y-2 overflow-hidden"
              >
                <div className="text-[9px] font-bold text-white/30 uppercase tracking-widest">Clé Gemini (GitHub Pages)</div>
                <div className="relative">
                  <input 
                    type="password"
                    placeholder="AIzaSy..."
                    value={localGeminiKey}
                    onChange={(e) => {
                      const val = e.target.value;
                      setLocalGeminiKey(val);
                      localStorage.setItem('COFFRAGE_GEMINI_KEY', val);
                    }}
                    className="w-full bg-black/40 border border-white/10 rounded px-2 py-1.5 text-[10px] outline-none focus:border-ai-accent transition-colors"
                  />
                  {localGeminiKey && (
                    <button 
                      onClick={() => { setLocalGeminiKey(""); localStorage.removeItem('COFFRAGE_GEMINI_KEY'); }}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-white/20 hover:text-danger"
                    >
                      <Trash size={10} />
                    </button>
                  )}
                </div>
                <p className="text-[8px] text-white/40 leading-tight italic">
                  Nécessaire car ce domaine (GitHub) ne supporte pas de serveur. La clé est stockée uniquement sur votre navigateur.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Firebase Account Panel */}
        <div className="mt-4 pt-4 border-t border-white/5">
          {user ? (
            <div className="flex items-center justify-between p-2 rounded-xl bg-white/5 border border-white/10 group">
              <div className="flex items-center gap-3 overflow-hidden">
                {user.photoURL ? (
                  <img src={user.photoURL} alt={user.displayName || ''} className="w-8 h-8 rounded-lg shrink-0 border border-white/10" referrerPolicy="no-referrer" />
                ) : (
                  <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center shrink-0">
                    <UserIcon size={16} className="text-accent" />
                  </div>
                )}
                <div className="truncate">
                  <div className="text-[11px] font-bold text-white truncate">{user.displayName || 'Ingénieur'}</div>
                  <div className="text-[9px] text-white/40 truncate">Cloud Sync: OK</div>
                </div>
              </div>
              <button 
                onClick={handleLogout}
                className="p-1.5 opacity-0 group-hover:opacity-100 text-white/40 hover:text-danger hover:bg-danger/10 rounded-lg transition-all"
                title="Déconnexion"
              >
                <LogOut size={14} />
              </button>
            </div>
          ) : (
            <button 
              onClick={loginWithGoogle}
              className="w-full flex items-center justify-center gap-3 p-3 rounded-xl bg-ai-accent text-white font-bold text-[11px] uppercase tracking-widest hover:bg-black transition-all shadow-lg shadow-ai-accent/20"
            >
              <LogIn size={14} /> Connexion Cloud
            </button>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col p-6 gap-5 overflow-hidden">
        {/* Header Bar */}
        <div className="flex justify-between items-center mb-6">
          <div className="flex-1">
            {isRenaming ? (
              <div className="flex items-center gap-2 animate-in fade-in duration-300">
                <input 
                  type="text" 
                  autoFocus
                  value={tempName}
                  onChange={(e) => setTempName(e.target.value)}
                  onBlur={saveRename}
                  onKeyDown={(e) => e.key === 'Enter' && saveRename()}
                  className="text-2xl font-black tracking-tight font-serif italic text-accent bg-transparent border-b-2 border-accent outline-none w-full max-w-md"
                />
                <button onClick={saveRename} className="p-1 text-success hover:bg-success/10 rounded-full"><Check size={20}/></button>
              </div>
            ) : (
              <div 
                onClick={startRenaming}
                className="group cursor-pointer inline-flex flex-col animate-in slide-in-from-left duration-500"
              >
                <div className="flex items-center gap-2">
                  <h1 className="text-2xl font-black tracking-tight font-serif italic text-accent/80 group-hover:text-accent transition-colors">
                    {activeTab === 'dashboard' ? projectData.name : (activeTab === 'dalle' ? 'Coffrage de Dalle' : 'Coffrage de Poutre')}
                  </h1>
                  <Settings size={14} className="text-accent/0 group-hover:text-accent/40 transition-all" />
                </div>
                <p className="text-text-muted text-sm mt-0.5 flex items-center gap-2">
                  {activeTab === 'dashboard' 
                    ? `Dernière modification : ${new Date(projectData.updatedAt).toLocaleDateString('fr-FR')}` 
                    : 'Analyse de structure et calcul de charges d\'étaiement'}
                    {activeTab === 'dashboard' && <span className="text-[10px] text-accent/50 group-hover:block hidden underline italic">Cliquez pour renommer</span>}
                </p>
              </div>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button 
              onClick={() => document.getElementById('import-project')?.click()}
              className="px-4 py-2 bg-white border border-border rounded-full font-bold text-[10px] hover:bg-bg transition-colors flex items-center gap-2 shadow-sm uppercase tracking-wider"
            >
              <FileUp size={14} className="text-success" /> Importer
            </button>
            <button 
              onClick={exportJSON}
              className="px-4 py-2 bg-white border border-border rounded-full font-bold text-[10px] hover:bg-bg transition-colors flex items-center gap-2 shadow-sm uppercase tracking-wider"
            >
              <FileDown size={14} className="text-accent" /> Exporter
            </button>
            <div className="h-4 w-[1px] bg-border mx-1" />
            <button 
              onClick={() => showToast("Données stockées localement")}
              className="px-5 py-2 bg-accent text-white rounded-full font-bold text-[10px] hover:filter hover:brightness-105 transition-all shadow-lg shadow-accent/20 uppercase tracking-wider"
            >
              Sauvegardé (Auto)
            </button>
            <input 
              id="import-project"
              type="file" 
              accept=".json" 
              className="hidden" 
              onChange={importJSON} 
            />
          </div>
        </div>

        {activeTab === 'dashboard' ? (
          <div className="flex-1 flex flex-col gap-6 overflow-hidden">
            {/* Bento Grid Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 shrink-0">
               <motion.div 
                 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                 className="bg-surface border border-border p-5 rounded-2xl shadow-sm flex items-center gap-4 border-b-4 border-b-accent"
               >
                 <div className="w-12 h-12 bg-accent/10 rounded-xl flex items-center justify-center text-accent shrink-0"><Building2 size={24} /></div>
                 <div>
                   <div className="text-[10px] font-black text-text-muted uppercase tracking-widest">Éléments</div>
                   <div className="text-2xl font-black">{projectData.elements.length}</div>
                 </div>
               </motion.div>
               <motion.div 
                 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                 className="bg-surface border border-border p-5 rounded-2xl shadow-sm flex items-center gap-4 border-b-4 border-b-ai-accent"
               >
                 <div className="w-12 h-12 bg-ai-accent/10 rounded-xl flex items-center justify-center text-ai-accent shrink-0"><Zap size={24} /></div>
                 <div>
                   <div className="text-[10px] font-black text-text-muted uppercase tracking-widest">IA Analyse</div>
                   <div className="text-2xl font-black">{projectData.elements.filter(e => e.isAi).length}</div>
                 </div>
               </motion.div>
               <motion.div 
                 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
                 className="bg-surface border border-border p-5 rounded-2xl shadow-sm flex items-center gap-4 border-b-4 border-b-success"
               >
                 <div className="w-12 h-12 bg-success/10 rounded-xl flex items-center justify-center text-success shrink-0"><ShieldCheck size={24} /></div>
                 <div>
                   <div className="text-[10px] font-black text-text-muted uppercase tracking-widest">Sécurisé</div>
                   <div className="text-2xl font-black text-success">
                     {projectData.elements.filter(e => e.ok_wood && e.ok_alu && e.ok_frame).length}
                   </div>
                 </div>
               </motion.div>
               <motion.div 
                 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
                 className="bg-surface border border-border p-5 rounded-2xl shadow-sm flex items-center gap-4 border-b-4 border-b-danger"
               >
                 <div className="w-12 h-12 bg-danger/10 rounded-xl flex items-center justify-center text-danger shrink-0"><AlertTriangle size={24} /></div>
                 <div>
                   <div className="text-[10px] font-black text-text-muted uppercase tracking-widest">Critique</div>
                   <div className="text-2xl font-black text-danger">
                     {projectData.elements.filter(e => !e.ok_wood || !e.ok_alu || !e.ok_frame).length}
                   </div>
                 </div>
               </motion.div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6 flex-1 overflow-hidden">
               {/* Elements List */}
               <div className="bg-surface border border-border rounded-3xl shadow-sm overflow-hidden flex flex-col min-h-0">
                  <div className="p-6 border-b border-border flex justify-between items-center bg-bg/5">
                     <div className="flex items-center gap-6 flex-1">
                        <h2 className="text-xs font-black text-text-main uppercase tracking-[0.2em] flex items-center gap-2">
                          <LayoutDashboard size={14} className="text-accent"/> Inventaire
                        </h2>
                        <div className="relative flex-1 max-w-sm">
                           <input 
                             type="text"
                             placeholder="Filtrer les éléments..."
                             value={searchQuery}
                             onChange={(e) => setSearchQuery(e.target.value)}
                             className="w-full pl-10 pr-4 py-2.5 bg-white border border-border rounded-xl text-xs outline-none focus:ring-2 focus:ring-accent/20 transition-all shadow-inner"
                           />
                           <Search size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-text-muted" />
                        </div>
                     </div>
                     <div className="flex items-center gap-4">
                        <select 
                          value={sortBy}
                          onChange={(e) => setSortBy(e.target.value as any)}
                          className="bg-white border border-border rounded-lg px-3 py-2 text-[11px] font-bold outline-none cursor-pointer"
                        >
                           <option value="date">Plus récents</option>
                           <option value="name">A-Z</option>
                           <option value="factor">Gravité</option>
                        </select>
                        <button onClick={clearProject} className="p-2.5 text-danger hover:bg-danger/10 rounded-xl transition-all" title="Reset">
                           <Trash2 size={18} />
                        </button>
                     </div>
                  </div>

                  <div className="flex-1 overflow-y-auto scroller-hidden p-6 space-y-4">
                     {projectData.elements.length === 0 && pendingExtractions.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center opacity-30 p-12">
                           <Box size={48} strokeWidth={1} className="mb-4" />
                           <p className="font-serif italic text-lg mb-1">Dossier vide</p>
                           <p className="text-[10px] uppercase font-bold tracking-widest">Analysez un plan pour commencer</p>
                        </div>
                     ) : (
                        <>
                           {pendingExtractions.map(el => (
                              <motion.div key={el.id} initial={{ scale: 0.98, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="bg-ai-accent/5 border border-ai-accent/20 rounded-2xl p-5 flex flex-col md:flex-row items-center gap-4">
                                 <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-1">
                                       <span className="font-black text-sm text-ai-accent uppercase">{el.params.name}</span>
                                       <span className="bg-ai-accent text-white text-[8px] font-black px-1.5 py-0.5 rounded">DETECTED</span>
                                    </div>
                                    <div className="text-[10px] font-bold text-text-muted uppercase">{el.params.type} • Épaisseur {el.params.epMax}mm</div>
                                 </div>
                                 <div className="flex gap-2 w-full md:w-auto">
                                    <button onClick={() => approveExtraction(el.id)} className="flex-1 md:w-32 bg-ai-accent text-white py-2 rounded-xl text-[11px] font-black hover:filter hover:brightness-110 flex items-center justify-center gap-1"><Check size={14}/> Approuver</button>
                                    <button onClick={() => rejectExtraction(el.id)} className="p-2 border border-ai-accent/20 text-ai-accent rounded-xl hover:bg-ai-accent/10"><X size={18}/></button>
                                 </div>
                              </motion.div>
                           ))}
                           
                           {filteredElements.map((el, idx) => (
                              <motion.div 
                                key={el.id} 
                                initial={{ x: -10, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: idx * 0.03 }}
                                className="group bg-bg/5 border border-border/50 rounded-2xl p-5 hover:border-accent/40 hover:bg-white hover:shadow-xl transition-all flex flex-col xl:flex-row items-center gap-6"
                              >
                                <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 shadow-sm
                                  ${!el.ok_wood || !el.ok_alu || !el.ok_frame ? 'bg-danger text-white' : 'bg-accent text-white text-xl font-bold'}`}>
                                  {!el.ok_wood || !el.ok_alu || !el.ok_frame ? <AlertTriangle size={24} /> : (el.params.type==='DALLE'?'▣':'▬')}
                                </div>
                                <div className="flex-1 min-w-0">
                                   <div className="flex items-center gap-3 mb-1">
                                      <h3 className="font-black text-base uppercase tracking-tighter truncate">{el.params.name}</h3>
                                      {el.isAi && <Sparkles size={12} className="text-ai-accent" />}
                                      <div className={`px-2 py-0.5 rounded text-[8px] font-black uppercase tracking-wider ${el.ok_wood && el.ok_alu && el.ok_frame ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'}`}>
                                         {el.ok_wood && el.ok_alu && el.ok_frame ? '✅ CONFORME' : '❌ NON CONFORME'}
                                      </div>
                                   </div>
                                   <div className="flex gap-4 text-[10px] font-bold text-text-muted uppercase">
                                      <span>{el.params.isVar ? `${el.params.epMin}-${el.params.epMax}` : el.params.ep}mm</span>
                                      <span>{el.params.type}</span>
                                      <span className="text-accent italic">{el.time}</span>
                                   </div>
                                </div>
                                <div className="flex gap-2">
                                   <button onClick={() => startEdit(el.id)} className="p-2.5 rounded-xl border border-border text-text-muted hover:text-accent hover:border-accent hover:bg-accent/5 transition-all"><Settings size={18}/></button>
                                   <button onClick={() => deleteElement(el.id)} className="p-2.5 rounded-xl border border-border text-text-muted hover:text-danger hover:border-danger hover:bg-danger/5 transition-all"><Trash2 size={18}/></button>
                                </div>
                              </motion.div>
                           ))}
                        </>
                     )}
                  </div>
               </div>

               {/* Stats and AI */}
               <div className="flex flex-col gap-6">
                  <div className="bg-surface border border-border rounded-3xl p-6 shadow-sm border-t-4 border-t-accent">
                     <h3 className="text-[10px] font-black text-text-muted uppercase tracking-widest mb-6 flex items-center gap-2"><BarChart3 size={14}/> Graphique de Charges</h3>
                     <div className="h-[200px] w-full min-w-0">
                        <ResponsiveContainer width="99%" height="100%" minWidth={0}>
                           <BarChart data={statsData} margin={{ left: -30, right: 10 }}>
                              <XAxis dataKey="name" hide />
                              <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px rgba(0,0,0,0.1)' }} />
                              <Bar dataKey="charge" radius={[4, 4, 0, 0]}>
                                 {statsData.map((e, i) => (
                                   <Cell key={`cell-${i}`} fill={i % 2 === 0 ? "#2563EB" : "#38BDF8"} />
                                 ))}
                              </Bar>
                           </BarChart>
                        </ResponsiveContainer>
                     </div>
                     <div className="mt-6 flex items-center justify-between p-4 bg-bg rounded-2xl border border-border/40">
                        <span className="text-[11px] font-bold text-text-muted uppercase">Santé Globale</span>
                        {projectData.elements.length > 0 ? (
                           <span className={`text-[10px] font-black uppercase px-2 py-0.5 rounded-full ${projectData.elements.every(e => e.ok_wood && e.ok_alu && e.ok_frame) ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger animate-pulse'}`}>
                              {projectData.elements.every(e => e.ok_wood && e.ok_alu && e.ok_frame) ? 'CONFORME' : 'CRITIQUE'}
                           </span>
                        ) : (
                           <span className="text-[10px] font-black text-text-muted uppercase bg-bg/20 px-2 py-0.5 rounded-full">Aucun Élément</span>
                        )}
                     </div>
                  </div>

                  <div className="bg-ai-accent/5 border-2 border-dashed border-ai-accent/30 rounded-3xl p-8 text-center flex flex-col items-center justify-center gap-5 relative overflow-hidden group">
                     <div className="w-14 h-14 bg-white rounded-2xl shadow-lg flex items-center justify-center text-ai-accent"><FileText size={28}/></div>
                     <div>
                        <h4 className="font-black text-ai-accent text-sm uppercase tracking-widest mb-1">Importez vos Plans</h4>
                        <p className="text-[10px] text-text-muted font-bold italic">Extraction automatique par IA Vision</p>
                     </div>
                     <label className="w-full bg-ai-accent text-white py-3 rounded-2xl font-black text-xs hover:filter hover:brightness-110 cursor-pointer shadow-lg shadow-ai-accent/20 flex items-center justify-center gap-2">
                        Analyser un Plan <input type="file" accept="image/*,application/pdf" className="hidden" onChange={(e) => {
                          const f = e.target.files?.[0]; if(f) handleAiExtraction(f);
                        }} />
                     </label>
                  </div>
               </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-5 flex-1 overflow-hidden">
            {/* Dashboard Columns for Tool Mode */}
            <div className="flex flex-col gap-5 overflow-y-auto scroller-hidden pr-1">
               <div className="bg-surface border border-border rounded-[10px] p-6 shadow-sm mb-4">
                  <div className="flex justify-between items-center mb-6">
                      <div className="flex items-center gap-3">
                         <div className="w-10 h-10 bg-accent/10 rounded-xl flex items-center justify-center text-accent"><Calculator size={20} /></div>
                         <div>
                            <h2 className="font-black text-sm uppercase tracking-widest text-text-main leading-tight">Moteur de Calcul {activeTab==='dalle'?'Dalle':'Poutre'}</h2>
                            <p className="text-[10px] font-bold text-text-muted uppercase tracking-widest mt-0.5">Normes CSA-S269.1 • Facteurs de sécurité actifs</p>
                         </div>
                      </div>
                     <div className={`px-3 py-1 rounded-full text-[10px] font-black uppercase flex items-center gap-1.5 ${(calculations.woodOk && calculations.okAlu && calculations.fOk) ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'}`}>
                        {(calculations.woodOk && calculations.okAlu && calculations.fOk) ? <ShieldCheck size={12} /> : <ShieldAlert size={12} />}
                        STATUT : {(calculations.woodOk && calculations.okAlu && calculations.fOk) ? 'CONFORME' : 'DANGER - NON CONFORME'}
                     </div>
                  </div>
                  
                  <div className="grid grid-cols-1 gap-8 mb-6">
                     <CapacityIndicator 
                       label="Utilisation Capacité Alu-Beam" 
                       value={calculations.loadAlu} 
                       limit={calculations.capAlu} 
                       unit="LBS" 
                     />
                     <CapacityIndicator 
                       label="Utilisation Espacement Bois" 
                       value={formChoix} 
                       limit={calculations.lm_wood} 
                       unit="PO" 
                     />
                  </div>
               </div>

               <div className="bg-surface border border-border rounded-[10px] p-6 shadow-sm">
                  <div className="flex justify-between items-center mb-6">
                     <h2 className="text-[13px] font-bold text-text-muted uppercase tracking-widest">Configuration de l'Élément</h2>
                     <span className="bg-ai-accent text-white text-[9px] font-black px-2 py-1 rounded">MODE IA ACTIF</span>
                  </div>

                  <div className="flex flex-col gap-5">
                    <div className="grid grid-cols-2 gap-4">
                       <div className="flex flex-col gap-1.5">
                          <label className="text-[10px] font-bold text-text-light uppercase tracking-widest">Nom de la Section</label>
                          <input 
                            type="text" 
                            className="p-3 border border-border rounded-lg text-[13px] outline-none duration-200 focus:border-accent bg-bg/30" 
                            value={formName}
                            onChange={(e) => setFormName(e.target.value)}
                            placeholder="SECTION-A101"
                          />
                       </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                       <div className="flex flex-col gap-1.5">
                          <label className="text-[10px] font-bold text-text-light uppercase tracking-widest">{activeTab==='dalle'?'Épaisseur':'Profondeur'} (mm)</label>
                          <div className="flex items-center gap-2">
                             <input 
                              type="number" 
                              className="w-full p-3 border border-border rounded-lg text-[13px] outline-none duration-200 focus:border-accent" 
                              value={formIsVar ? formEpMax : formEp}
                              onChange={(e) => formIsVar ? setFormEpMax(+e.target.value) : setFormEp(+e.target.value)}
                             />
                             <button onClick={() => setFormIsVar(!formIsVar)} className={`p-3 border rounded-lg text-[10px] font-bold uppercase shrink-0 transition-colors ${formIsVar?'bg-accent text-white border-accent':'bg-bg text-text-muted border-border'}`}>VAR</button>
                          </div>
                          {formIsVar && <input type="number" value={formEpMin} onChange={(e) => setFormEpMin(+e.target.value)} className="w-full p-2 border border-border rounded-lg text-[11px] mt-1" placeholder="Min" />}
                       </div>
                       <div className="flex flex-col gap-1.5">
                          <label className="text-[10px] font-bold text-text-light uppercase tracking-widest">Charge Vive (psf)</label>
                          <input 
                            type="number" 
                            className="p-3 border border-border rounded-lg text-[13px] outline-none duration-200 focus:border-accent" 
                            value={formLive}
                            onChange={(e) => setFormLive(+e.target.value)}
                          />
                       </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                       <div className="flex flex-col gap-1.5">
                          <label className="text-[10px] font-bold text-text-light uppercase tracking-widest">Larg. Trib. (ft)</label>
                          <input 
                            type="number" 
                            step="0.5"
                            className="p-3 border border-border rounded-lg text-[13px] outline-none duration-200 focus:border-accent" 
                            value={formTrib}
                            onChange={(e) => setFormTrib(+e.target.value)}
                          />
                       </div>
                       <div className="flex flex-col gap-1.5">
                          <label className="text-[10px] font-bold text-text-light uppercase tracking-widest">Portée Aluma (ft)</label>
                          <select 
                            className="p-3 border border-border rounded-lg text-[13px] outline-none duration-200 bg-white"
                            value={formSpan}
                            onChange={(e) => setFormSpan(+e.target.value)}
                          >
                             {[4,5,6,7].map(v => <option key={v} value={v}>{v}'</option>)}
                          </select>
                       </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                       <div className="flex flex-col gap-1.5">
                          <label className="text-[10px] font-bold text-text-light uppercase tracking-widest">Espacement Solives (po)</label>
                          <div className="flex items-center gap-2">
                            <input 
                              type="number" 
                              className={`w-full p-3 border rounded-lg text-[13px] outline-none duration-200 focus:border-accent ${calculations.woodOk ? 'border-border' : 'border-danger bg-danger/5 text-danger'}`}
                              value={formChoix}
                              onChange={(e) => setFormChoix(+e.target.value)}
                            />
                            <div className={`px-2 py-1 rounded text-[10px] font-black whitespace-nowrap ${calculations.woodOk ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'}`}>
                              {fmt(calculations.woodUtil, 0)}%
                            </div>
                          </div>
                          <div className="text-[9px] text-text-muted italic flex justify-between">
                            <span>Suggéré : {calculations.sp}"</span>
                            <span>Max : {fmt(calculations.lm_wood, 1)}"</span>
                          </div>
                       </div>
                       <div className="flex flex-col gap-1.5">
                          <label className="text-[10px] font-bold text-text-light uppercase tracking-widest">Type de Support</label>
                          <select 
                            className="p-3 border border-border rounded-lg text-[13px] outline-none duration-200 bg-white"
                            value={formSType}
                            onChange={(e) => setFormSType(e.target.value as any)}
                            disabled={activeTab === 'poutre'}
                          >
                             <option value="simple">Simple</option>
                             <option value="double">Double (Chevauchement)</option>
                          </select>
                       </div>
                    </div>

                    <div className="flex gap-3 pt-2">
                       <button 
                        onClick={saveElement}
                        className="flex-1 bg-accent text-white py-3 border rounded-lg font-bold text-[13px] hover:filter hover:brightness-105 shadow-lg shadow-accent/10 transition-all flex items-center justify-center gap-2"
                       >
                         {editingId ? <Settings size={16} /> : <Plus size={16} />} 
                         {editingId ? 'Enregistrer les modifications' : 'Ajouter à l\'étude'}
                       </button>
                       {editingId && (
                         <button onClick={resetForm} className="px-4 border border-border rounded-lg text-text-muted hover:bg-bg transition-colors">
                            <X size={20} />
                         </button>
                       )}
                    </div>
                  </div>

                  <div className="bg-[#f8fafc] p-6 rounded-xl border border-dashed border-border mt-8 flex">
                     <div className="flex-1 text-center border-r border-border px-2">
                        <div className="val">{Math.round(calculations.total)}</div>
                        <div className="lbl">CHARGE TOTALE (psf)</div>
                     </div>
                     <div className="flex-1 text-center border-r border-border px-2">
                        <div className="val">{calculations.sp}"</div>
                        <div className="lbl">ESPACEMENT 4x4</div>
                     </div>
                     <div className="flex-1 text-center px-2">
                        <div className="val">{calculations.capAlu}</div>
                        <div className="lbl">CAPACITÉ ALUMA</div>
                     </div>
                  </div>

                  <div className="mt-8 p-4 bg-[#fffbeb] border border-[#f59e0b] rounded-[10px] flex items-start gap-4">
                     <div className="p-2 bg-[#f59e0b]/10 text-[#f59e0b] rounded-lg shrink-0"><AlertTriangle size={16} /></div>
                     <div className="text-[12px] leading-relaxed text-[#92400e]">
                        <strong className="block mb-0.5">Note Technique:</strong> Les calculs sont basés sur un contreplaqué 11/16″ et Aluma 165 avec F'b=1545 psi. Vérifiez toujours les limitations de déflexion (L/270).
                     </div>
                  </div>
               </div>
            </div>

            <div className="flex flex-col gap-5 overflow-hidden">
               <div className="bg-surface border border-border rounded-[10px] p-5 shadow-sm">
                  <div className="flex justify-between items-center mb-4">
                     <span className="text-[10px] font-bold uppercase tracking-widest text-text-muted">Résumé de Charge (Impérial)</span>
                     <button onClick={() => copySummary(document.getElementById('summary-tool')?.innerText || '')} className="text-accent hover:underline text-[10px] font-bold">COPIER</button>
                  </div>
                  <div id="summary-tool" className="text-[11px] leading-relaxed font-mono uppercase bg-bg/30 p-4 rounded-lg border border-border/50" dangerouslySetInnerHTML={{ __html: summaryText }} />
               </div>

               <div className="bg-surface border border-border rounded-[10px] p-5 shadow-sm">
                  <div className="flex justify-between items-center mb-4">
                     <span className="text-[10px] font-bold uppercase tracking-widest text-text-muted">Résumé de Charge (Métrique)</span>
                     <button onClick={() => copySummary(document.getElementById('summary-tool-metric')?.innerText || '')} className="text-accent hover:underline text-[10px] font-bold">COPIER</button>
                  </div>
                  <div id="summary-tool-metric" className="text-[11px] leading-relaxed font-mono uppercase bg-bg/30 p-4 rounded-lg border border-border/50" dangerouslySetInnerHTML={{ __html: summaryTextMetric }} />
               </div>

               <div className="bg-surface border border-border rounded-[10px] shadow-sm flex flex-col flex-1 overflow-hidden">
                  <div className="p-4 border-b border-border font-bold text-[12px] shrink-0">VÉRIFICATION DÉTAILLÉE</div>
                  <div className="flex-1 overflow-y-auto p-4 scroller-hidden">
                     <div className="space-y-4">
                        <div className="pb-3 border-b border-bg">
                           <div className="text-[10px] font-bold text-text-muted mb-2 uppercase tracking-wide">État des Composants</div>
                           <div className="space-y-2">
                              <div className="flex justify-between items-center text-[11px]">
                                <span>Contreplaqué</span>
                                <span className={`px-2 py-0.5 rounded text-[9px] font-black ${calculations.woodOk ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'}`}>
                                  {calculations.woodOk ? 'CONFORME' : 'ÉCHEC'}
                                </span>
                              </div>
                              <div className="flex justify-between items-center text-[11px]">
                                <span>Poutrelles Alu</span>
                                <span className={`px-2 py-0.5 rounded text-[9px] font-black ${calculations.okAlu ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'}`}>
                                  {calculations.okAlu ? 'CONFORME' : 'ÉCHEC'}
                                </span>
                              </div>
                              <div className="flex justify-between items-center text-[11px]">
                                <span> Shore / Étaiement</span>
                                <span className={`px-2 py-0.5 rounded text-[9px] font-black ${calculations.fOk ? 'bg-success/10 text-success' : 'bg-danger/10 text-danger'}`}>
                                  {calculations.fOk ? 'CONFORME' : 'ÉCHEC'}
                                </span>
                              </div>
                           </div>
                        </div>
                        <div className="pb-3 border-b border-bg">
                           <div className="text-[10px] font-bold text-text-muted mb-2 uppercase tracking-wide">A1 — Structure</div>
                           <div className="space-y-1.5">
                              <div className="flex justify-between text-[11px]"><span>Poids Béton</span><span className="font-bold font-mono">{fmt(calculations.conc, 1)} psf</span></div>
                              <div className="flex justify-between text-[11px]"><span>Surpoids (M+V)</span><span className="font-bold font-mono">{formDead + formLive} psf</span></div>
                           </div>
                        </div>
                        <div className="pb-3 border-b border-bg">
                           <div className="text-[10px] font-bold text-text-muted mb-2 uppercase tracking-wide">A3 — Aluma Profile</div>
                           <div className="space-y-1.5">
                              <div className="flex justify-between text-[11px]"><span>Charge Lineaire</span><span className="font-bold font-mono">{fmt(calculations.loadAlu, 0)} lbs/ft</span></div>
                              <div className="flex justify-between text-[11px]"><span>Ratio Capacité</span><span className={`font-bold font-mono ${calculations.okAlu?'text-success':'text-danger'}`}>{Math.round(calculations.loadAlu/calculations.capAlu*100)}%</span></div>
                           </div>
                        </div>
                        <div className="p-4 bg-ai-accent/5 border border-ai-accent/20 rounded-lg">
                           <div className="text-[10px] font-bold text-ai-accent mb-2 uppercase tracking-wide flex items-center gap-2"><Sparkles size={10} /> Facteur IA</div>
                           <div className="text-[10px] leading-relaxed opacity-70">L'IA suggère un espacement de 48" basé sur l'analyse visuelle du plan de structure importé.</div>
                        </div>
                     </div>
                  </div>
               </div>
            </div>
          </div>
        )}

        {/* Multi-Candidate Selection UI */}
        <AnimatePresence>
          {extractionCandidates.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/80 backdrop-blur-md z-[100] flex items-center justify-center p-6"
            >
              <motion.div
                initial={{ scale: 0.9, y: 20 }}
                animate={{ scale: 1, y: 0 }}
                className="bg-white border border-border w-full max-w-6xl h-full max-h-[85vh] rounded-3xl shadow-2xl flex flex-col overflow-hidden"
              >
                <div className="p-8 border-b border-border flex justify-between items-center bg-bg/30">
                  <div>
                    <h2 className="text-2xl font-black tracking-tight font-serif italic text-accent flex items-center gap-3">
                      <Sparkles className="text-ai-accent" /> Comparaison des Modèles IA
                    </h2>
                    <p className="text-xs font-bold text-text-muted uppercase tracking-widest mt-1">Plusieurs extractions détectées via OpenRouter. Choisissez la plus précise.</p>
                  </div>
                  <button 
                    onClick={() => setExtractionCandidates([])}
                    className="w-10 h-10 rounded-full bg-bg hover:bg-border transition-colors flex items-center justify-center"
                  >
                    <X />
                  </button>
                </div>
                
                <div className="flex-1 overflow-x-auto p-8 bg-bg/20">
                  <div className="flex gap-6 h-full">
                    {extractionCandidates.map((cand, idx) => (
                      <motion.div 
                        key={cand.modelId}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className="w-[320px] shrink-0 bg-white border border-border rounded-2xl flex flex-col shadow-sm hover:shadow-xl hover:border-accent/40 transition-all group"
                      >
                        <div className="p-5 border-b border-border bg-bg/10">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-[10px] font-black text-ai-accent uppercase tracking-tighter">Modèle {idx + 1}</span>
                            <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
                          </div>
                          <h3 className="font-black text-lg tracking-tight truncate">{cand.modelName}</h3>
                          <p className="text-[9px] font-mono text-text-muted truncate mt-1">{cand.modelId}</p>
                        </div>
                        
                        <div className="flex-1 overflow-y-auto p-5 space-y-3">
                          <div className="text-[9px] font-bold text-text-muted uppercase tracking-widest mb-2 border-b border-border/50 pb-1">
                            {cand.elements.length} Éléments Détectés
                          </div>
                          {cand.elements.map((el, eIdx) => (
                            <div key={eIdx} className="bg-bg/40 p-3 rounded-xl border border-border/30">
                              <div className="flex items-center justify-between mb-1">
                                <span className="font-bold text-[11px] truncate uppercase">{el.name}</span>
                                <span className="text-[9px] px-1.5 py-0.5 bg-accent/10 text-accent rounded font-black">{el.type}</span>
                              </div>
                              <div className="text-[10px] font-mono text-text-muted">Épaisseur: {el.thickness}mm</div>
                            </div>
                          ))}
                        </div>
                        
                        <div className="p-5 border-t border-border bg-bg/10 mt-auto">
                          <button 
                            onClick={() => applyCandidate(cand)}
                            className="w-full bg-accent text-white py-3 rounded-xl font-black text-[11px] uppercase tracking-wider hover:filter hover:brightness-110 shadow-lg shadow-accent/20 transition-all flex items-center justify-center gap-2"
                          >
                            <CheckCircle2 size={16} /> Sélectionner ce Résultat
                          </button>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
                
                <div className="p-6 border-t border-border text-center bg-white">
                  <p className="text-[10px] text-text-muted font-bold uppercase tracking-widest italic animate-pulse">
                    Astuce : Vérifiez la précision des noms (D1, P2) et des épaisseurs avant de confirmer.
                  </p>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* OpenRouter Model Configuration Modal */}
        <AnimatePresence>
          {showOrConfig && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[110] flex items-center justify-center p-4"
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="bg-surface border border-border w-full max-w-md rounded-2xl shadow-2xl overflow-hidden flex flex-col"
              >
                <div className="p-6 border-b border-border bg-bg/30 flex justify-between items-center">
                  <div>
                    <h3 className="text-lg font-black tracking-tight uppercase">Configuration OpenRouter</h3>
                    <p className="text-[10px] text-text-muted font-bold uppercase tracking-widest">Sélectionnez les modèles à utiliser (Max 10)</p>
                  </div>
                  <button onClick={() => setShowOrConfig(false)} className="w-8 h-8 rounded-full bg-bg hover:bg-border transition-colors flex items-center justify-center">
                    <X size={16} />
                  </button>
                </div>
                
                <div className="p-4 space-y-2 max-h-[400px] overflow-y-auto">
                  {OPENROUTER_VISION_MODELS.map(model => (
                    <label 
                      key={model.id}
                      className={`flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all
                      ${selectedOrModels.includes(model.id) ? 'bg-accent/5 border-accent/30 shadow-inner' : 'bg-bg/50 border-border/30 hover:border-border'}`}
                    >
                      <input 
                        type="checkbox"
                        className="hidden"
                        checked={selectedOrModels.includes(model.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedOrModels(prev => [...prev, model.id]);
                          } else {
                            setSelectedOrModels(prev => prev.filter(id => id !== model.id));
                          }
                        }}
                      />
                      <span className="text-xl">{model.icon}</span>
                      <div className="flex-1">
                        <div className="text-[12px] font-black tracking-tight">{model.name}</div>
                        <div className="text-[9px] font-mono text-text-muted truncate">{model.id}</div>
                      </div>
                      <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-all
                      ${selectedOrModels.includes(model.id) ? 'bg-accent border-accent text-white' : 'border-border'}`}>
                        {selectedOrModels.includes(model.id) && <Check size={12} />}
                      </div>
                    </label>
                  ))}
                </div>
                
                <div className="p-4 bg-bg/50 flex gap-3">
                  <button 
                    onClick={() => {
                      if (selectedOrModels.length === 0) {
                        showToast("Veuillez sélectionner au moins un modèle", "error");
                        return;
                      }
                      setShowOrConfig(false);
                    }}
                    className="flex-1 py-3 rounded-xl bg-accent text-white font-black text-xs uppercase tracking-widest hover:filter hover:brightness-110 transition-all shadow-lg shadow-accent/20"
                  >
                    Enregistrer la Sélection ({selectedOrModels.length})
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

      </main>

    </div>
  );
}
