


export function createPageUrl(pageName: string) {
    return '/digital/' + pageName.toLowerCase().replace(/ /g, '-');
}
