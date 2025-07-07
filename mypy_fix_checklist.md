# Comprehensive MyPy Fix Checklist

## Priority Levels:
- 🔴 **P0 - Critical**: Core modules that must be 100% type-safe
- 🟠 **P1 - High**: Service layer modules for production
- 🟡 **P2 - Medium**: Optional/demo files that can be improved later
- 🟢 **P3 - Low**: Example/demo files, non-critical for production

---

## 🔴 P0 - CRITICAL: Core Module Issues (PRODUCTION BLOCKERS)

### ✅ batch_persistence.py (COMPLETED)
- [x] Line 21: Fix unused "type: ignore" comment ✅
- [x] Line 24: Fix unused "type: ignore" comment ✅

### ✅ persistence.py (COMPLETED) 
- [x] Line 13: Fix unused "type: ignore" comment ✅
- [x] Line 20: Fix unused "type: ignore" comment ✅

### ✅ contract_persistence.py (COMPLETED)
- [x] Line 13: Fix unused "type: ignore" comment ✅
- [x] Line 58: Fix unreachable statement issue ✅
- [x] Line 339: Fix unsupported target for indexed assignment ("object") ✅
- [x] Line 386: Fix unused "type: ignore" comment ✅

### ✅ neural_symbolic_integration.py (COMPLETED)
- [x] Line 36: Fix unused "type: ignore" comment ✅
- [x] Line 236: Fix subclass incompatibility (list[float] and str/bytes) ✅
- [x] Line 567: Fix unused "type: ignore" comment ✅

---

## 🟠 P1 - HIGH: Service Layer Issues (PRODUCTION IMPORTANT)

### ✅ batch_service.py (MOSTLY COMPLETED)
- [x] Line 96: Add return type annotation ✅
- [x] Line 111: Add return type annotation ✅
- [x] Line 151: Add return type annotation ✅
- [x] Line 188: Add return type annotation ✅
- [x] Line 212: Add return type annotation ✅
- [ ] Line 229: Add return type annotation
- [ ] Line 259: Add return type annotation
- [ ] Line 282: Add return type annotation
- [ ] Line 311: Add return type annotation
- [ ] Line 350: Add return type annotation
- [ ] Line 362: Add return type annotation
- [ ] Line 420: Add return type annotation

### working_service_layer.py
- [ ] Line 82: Add return type annotation
- [ ] Line 162: Add return type annotation
- [ ] Line 178: Add return type annotation
- [ ] Line 187: Fix union-attr issue (None | EnhancedHybridRegistry)
- [ ] Line 192: Fix union-attr issue (None | ContractEnhancedPersistenceManager)
- [ ] Line 242: Add return type annotation
- [ ] Line 270: Add return type annotation
- [ ] Line 304: Add return type annotation
- [ ] Line 346: Add return type annotation
- [ ] Line 376: Add return type annotation
- [ ] Line 401: Add return type annotation
- [ ] Line 431: Add return type annotation
- [ ] Line 457: Add return type annotation
- [ ] Line 474: Add return type annotation (-> None)
- [ ] Line 486: Fix call to untyped function

### service_layer.py
- [ ] Line 72: Add type annotation
- [ ] Line 111: Add type annotation  
- [ ] Line 179: Add return type annotation
- [ ] Line 242: Fix call to untyped function
- [ ] Line 317: Add return type annotation
- [ ] Line 416: Fix argument type mismatch (Concept vs FrameAwareConcept)
- [ ] Line 746: Add type annotation for "instances"
- [ ] Line 887: Add return type annotation (-> None)
- [ ] Line 890: Add return type annotation
- [ ] Line 894: Add return type annotation
- [ ] Line 898: Add return type annotation
- [ ] Line 901: Add return type annotation
- [ ] Line 910: Fix call to untyped function
- [ ] Line 914: Add return type annotation
- [ ] Line 975: Add return type annotation
- [ ] Line 986: Fix undefined name "get_or_create_batch_manager"
- [ ] Line 1046: Add return type annotation
- [ ] Line 1062: Add return type annotation
- [ ] Lines 1070-1072: Fix union-attr issues (None | EnhancedHybridRegistry)
- [ ] Lines 1077, 1084-1085: Fix union-attr issues (None | BatchPersistenceManager)
- [ ] Line 1103: Add return type annotation
- [ ] Line 1134: Add return type annotation
- [ ] Line 1151: Add return type annotation (-> None)
- [ ] Line 1163: Fix call to untyped function
- [ ] Line 1166: Add return type annotation

### neural_symbolic_service.py
- [ ] Line 210: Add type annotation for "active_training_jobs"
- [ ] Line 211: Add type annotation for "completed_jobs"
- [ ] Line 259: Add return type annotation
- [ ] Line 351: Add type annotation for "axioms"
- [ ] Line 414: Add return type annotation
- [ ] Line 464: Add return type annotation
- [ ] Line 561: Add type annotation
- [ ] Lines 569, 594, 602, 615, 628: Fix untyped decorators
- [ ] Line 629: Add return type annotation

### main.py
- [ ] Line 30: Add return type annotation
- [ ] Line 62: Add return type annotation
- [ ] Line 68: Add return type annotation
- [ ] Line 74: Add return type annotation
- [ ] Line 99: Add return type annotation (-> None)
- [ ] Line 110: Fix call to untyped function

---

## 🟡 P2 - MEDIUM: Contract/Demo Issues (OPTIONAL)

### contract_enhanced_registry.py
- [ ] Line 52: Fix argument type mismatch in EnhancedHybridRegistry init
- [ ] Line 56: Add return type annotation (-> None)
- [ ] Lines 71-78: Fix icontract decorator argument types
- [ ] Lines 108-117: Fix icontract decorator argument types  
- [ ] Line 141: Fix list.items() attribute error
- [ ] Lines 153-164: Fix icontract decorator argument types
- [ ] Line 170: Fix Optional typing issue
- [ ] Line 181: Fix unreachable statement
- [ ] Line 190: Add type annotation for "structured_completions"
- [ ] Lines 202-209: Fix icontract decorator argument types
- [ ] Line 216: Add type parameters for generic tuple
- [ ] Line 250: Add return type annotation (-> None)
- [ ] Lines 273, 284, 295: Fix FrameAwareConcept.concept_id attribute
- [ ] Lines 274, 285, 296, 314, 325: Fix undefined "DpcontractsException"
- [ ] Line 343: Fix call to untyped function

### icontract_demo.py
- [ ] Line 79: Add return type annotation (-> None)
- [ ] Line 83: Add type annotation
- [ ] Line 88: Add type annotation
- [ ] Line 97: Add type annotation
- [ ] Line 112: Fix BaseRegistry type assignment
- [ ] Line 125: Add type annotation
- [ ] Line 192: Fix list.items() attribute error
- [ ] Line 232: Add type annotation for "structured_completions"
- [ ] Line 254: Add return type annotation (-> None)
- [ ] Line 272: Fix call to untyped function
- [ ] Line 390: Fix call to untyped function

---

## 🟢 P3 - LOW: Example/Demo Files (NON-CRITICAL)

### Root Level Scripts
- [ ] smt01.py: Lines 7, 65 - Add annotations and fix calls
- [ ] ltn01.py: Lines 56, 60, 64, 69 - Add type annotations
- [ ] explore_abstractions.py: Lines 10, 39, 73, 117, 157, 202, 254, 258-263, 269 - Add annotations

### Example Files
- [ ] persistence_examples_overview.py: Multiple missing annotations
- [ ] persistence_strategy_example.py: Multiple missing annotations
- [ ] multi_format_persistence_example.py: Multiple missing annotations

---

## 🎯 Action Plan

### Phase 1: Complete Critical Core Modules (P0) ⚡
1. Fix remaining contract_persistence.py issues
2. Fix remaining neural_symbolic_integration.py issues
3. Ensure 100% core type safety

### Phase 2: Service Layer Production Readiness (P1) 🚀
1. Add all missing return type annotations
2. Fix union-attr issues with proper null checks
3. Resolve argument type mismatches
4. Fix undefined name errors

### Phase 3: Optional Improvements (P2/P3) ✨
1. Contract/demo file improvements
2. Example file annotations (as time permits)

---

## Progress Tracking

### ✅ Completed (Core P0)
- batch_persistence.py: 2/2 issues fixed ✅
- persistence.py: 2/2 issues fixed ✅
- contract_persistence.py: 4/4 issues fixed ✅
- neural_symbolic_integration.py: 3/3 issues fixed ✅

**🎉 ALL P0 CRITICAL ISSUES COMPLETED - CORE IS 100% TYPE-SAFE! ✅**

### 🔄 In Progress
- Starting with contract_persistence.py critical fixes
- Moving to neural_symbolic_integration.py
- Then proceeding to service layer

### 📊 Statistics
- **Total Issues**: 240 errors
- **P0 Critical**: ~10 issues 
- **P1 High**: ~80 issues
- **P2 Medium**: ~60 issues  
- **P3 Low**: ~90 issues

**Target**: 100% type safety for P0/P1 (production-critical code)
