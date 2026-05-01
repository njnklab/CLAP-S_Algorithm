export type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends Record<string, any> ? DeepPartial<T[K]> : T[K];
};

function isPlainObject(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function cloneDeep<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((item) => cloneDeep(item)) as unknown as T;
  }

  if (isPlainObject(value)) {
    const result: Record<string, any> = {};
    Object.entries(value).forEach(([key, val]) => {
      result[key] = cloneDeep(val);
    });
    return result as T;
  }

  return value;
}

function mergeRecursive<T>(target: T, source?: DeepPartial<T>): T {
  if (!source) return target;
  const output: any = Array.isArray(target) ? [...(target as any[])] : { ...(target as any) };

  Object.entries(source).forEach(([key, value]) => {
    if (isPlainObject(value)) {
      const baseValue = (target as any)[key];
      output[key] = mergeRecursive(isPlainObject(baseValue) ? baseValue : {}, value);
    } else if (Array.isArray(value)) {
      output[key] = value.map((item) => cloneDeep(item));
    } else {
      output[key] = value;
    }
  });

  return output as T;
}

export function deepMerge<T>(base: T, ...sources: DeepPartial<T>[]): T {
  return sources.reduce<T>((acc, source) => mergeRecursive(acc, source), cloneDeep(base));
}

export function setByPath<T>(object: DeepPartial<T>, path: string, value: unknown): DeepPartial<T> {
  const clone = cloneDeep(object ?? ({} as DeepPartial<T>));
  const segments = path.split(".");
  let current: any = clone;

  segments.forEach((segment, index) => {
    if (index === segments.length - 1) {
      current[segment] = value;
      return;
    }
    if (!isPlainObject(current[segment])) {
      current[segment] = {};
    }
    current = current[segment];
  });

  return clone;
}
