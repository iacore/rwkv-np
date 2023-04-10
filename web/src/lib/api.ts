function currentWsProtoAuthority(): string {
  let url = new URL(import.meta.url)
  let proto
  switch (url.protocol) {
    case "http:":
      proto = "ws:"
      break
    case "https:":
      proto = "wss:"
      break
    default:
      throw new Error(`Invalid protocol ${url.protocol}`)
  }
  return `${proto}//${url.host}`
}

export const API_AUTHORITY =
  import.meta.env.VITE_API_AUTHORITY ?? currentWsProtoAuthority()
